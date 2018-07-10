import torch
import torchvision
import torchvision.transforms as transforms
from functools import reduce
from operator import mul
from util.network import *
from util.kmeans import *
import os
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class deployer(object):

	def __init__(self, retrained_model_CR, clusters_num_list, sparsity_p, visualize_filters, visualize_layers_index):

		self.workers = 4 # number of workers for dataloader
		self.seed = 1000 # seed for k-means
		self.bitwidth = 32 # default 32 bit float
		self.test_batch_size = 100 # testing batch size
		self.retrained_model_CR = retrained_model_CR # load re-trained model trained on specific compression ratio
		self.clusters_num_list = clusters_num_list # layer-wise k
		self.sparsity_p = sparsity_p # layer-wise sparsity p
		self.visualize_filters = visualize_filters # visualize filters
		self.visualize_layers_index = visualize_layers_index # layer index of visualized filters

	def get_samples_num_list(self, conv_layers):

		samples_num_list = []

		for i, layer in enumerate(conv_layers):

			weight = layer.weight.data

			filters_num = weight.shape[0]
			filters_channel = weight.shape[1]
			filters_size = weight.shape[2]
			samples_num = filters_num * filters_channel * filters_size

			samples_num_list.append(samples_num)

		return samples_num_list

	def get_memory(self):

		self.all_memory = 0
		self.layer_wise_memory = []

		for i, layer in enumerate(self.conv_layers):

			weight = layer.weight.data
			bias = layer.bias.data

			filters_num = weight.shape[0]
			filters_channel = weight.shape[1]
			filters_size = weight.shape[2]
			samples_num = filters_num * filters_channel * filters_size

			memory_weight = reduce(mul, weight.shape) * self.bitwidth
			memory_bias = reduce(mul, bias.shape) * self.bitwidth
			self.all_memory += memory_weight + memory_bias
			self.layer_wise_memory.append(memory_weight + memory_bias)

		for i, layer in enumerate(self.linear_layers):

			weight = layer.weight.data
			bias = layer.bias.data

			memory_weight = reduce(mul, weight.shape) * self.bitwidth
			memory_bias = reduce(mul, bias.shape) * self.bitwidth
			self.all_memory += memory_weight + memory_bias
			self.layer_wise_memory.append(memory_weight + memory_bias)

			self.memory_percent_layer = [a / self.all_memory * 100 for a in self.layer_wise_memory]

	def get_compressed_memory(self):

		self.all_memory_compressed = 0
		self.layer_wise_memory_compressed = []

		for i, layer in enumerate(self.conv_layers + self.linear_layers):

			if layer in self.conv_layers:

				weight = layer.weight.data.cpu().numpy()
				bias = layer.bias.data.cpu().numpy()
				filters_num = weight.shape[0]
				filters_channel = weight.shape[1]
				filters_size = weight.shape[2]
				samples_num = filters_num * filters_channel * filters_size
				feature_dim = filters_size
				weight_reshape = weight.reshape(-1, filters_size)
				unique, counts = np.unique(weight_reshape, return_counts=True, axis=0)
				all_count = weight_reshape.shape[0]
				prob = counts / all_count
				huffman_length = np.sum([- p * np.log2(p) for p in prob])

				memory_weight = self.clusters_num_list[i] * feature_dim * self.bitwidth + samples_num * huffman_length
				memory_bias = reduce(mul, bias.shape) * self.bitwidth

				self.all_memory_compressed += memory_weight + memory_bias
				self.layer_wise_memory_compressed.append(memory_weight + memory_bias)

			if layer in self.linear_layers:

				weight = layer.weight.data
				bias = layer.bias.data

				memory_weight = reduce(mul, weight.shape) * self.bitwidth
				memory_bias = reduce(mul, bias.shape) * self.bitwidth

				self.all_memory_compressed += memory_weight + memory_bias
				self.layer_wise_memory_compressed.append(memory_weight + memory_bias)

		self.compressed_memory_percent_layer = [a / self.all_memory_compressed * 100 for a in self.layer_wise_memory_compressed]
		self.compression_ratio = self.all_memory / self.all_memory_compressed

	def seek_layers(self, model):

		self.all_layers = []
		self.conv_layers = []
		self.linear_layers = []
		self.bn_layers = []

		for keys, values in model._modules.items():
			self.seek_layers_base(values)

	def seek_layers_base(self, values):

		if isinstance(values, nn.modules.conv.Conv2d) or isinstance(values, nn.modules.BatchNorm2d) or isinstance(values, nn.modules.Linear):
			self.all_layers.append(values)
			if isinstance(values, nn.modules.conv.Conv2d):
				self.conv_layers.append(values)
			elif isinstance(values, nn.modules.Linear):
				self.linear_layers.append(values)
			elif isinstance(values, nn.modules.BatchNorm2d):
				self.bn_layers.append(values)
		else:
			if isinstance(values, nn.Sequential):
				for items in values:
					self.seek_layers_base(items)
			if isinstance(values, wide_basic):
				for items in values.get_all():
					self.seek_layers_base(items)

	def cluster_filters(self, weight, n_clusters, seed):

		# weight: cuda tensor
		filters_num = weight.shape[0]
		filters_channel = weight.shape[1]
		filters_size = weight.shape[2]

		weight_vector = weight.reshape(-1, filters_size)

		weight_vector_clustered = k_means_gpu(weight_vector.astype('float32'), n_clusters, verbosity=0, seed=seed,
															  gpu_id=0).astype('float32')

		# unique_count = np.unique(weight_vector_clustered, axis=0, return_counts=True)[1]
		# all_count = np.sum(unique_count)
		# prob = unique_count / all_count
		# huffman_length = np.sum([- p * np.log2(p) for p in prob])
		# print('Unique Count: {} Huffman Length: {}'.format(unique_count, huffman_length))

		weight_cube_clustered = weight_vector_clustered.reshape(filters_num, filters_channel,
																	filters_size, -1)

		mse = mean_squared_error(weight_vector, weight_vector_clustered)

		weight_compress = torch.from_numpy(weight_cube_clustered.astype('float32')).cuda()

		return weight_compress, mse

	def cluster_filters_sparsity(self, weight, n_clusters, ratio, seed):

		# weight: cuda tensor
		filters_num = weight.shape[0]
		filters_channel = weight.shape[1]
		filters_size = weight.shape[2]

		weight_vector = weight.reshape(-1, filters_size)

		weight_vector_clustered = k_means_gpu_sparsity(weight_vector.astype('float32'), n_clusters, ratio=ratio,
													   verbosity=0, seed=seed, gpu_id=0).astype('float32')
		# unique_count = np.unique(weight_vector_clustered, axis=0, return_counts=True)[1]
		# all_count = np.sum(unique_count)
		# prob = unique_count / all_count
		# huffman_length = np.sum([- p * np.log2(p) for p in prob])
		# print('Unique Count: {} Huffman Length: {}'.format(unique_count, huffman_length))

		weight_cube_clustered = weight_vector_clustered.reshape(filters_num, filters_channel, filters_size, -1)

		mse = mean_squared_error(weight_vector, weight_vector_clustered)

		weight_compress = torch.from_numpy(weight_cube_clustered.astype('float32')).cuda()

		return weight_compress, mse

	def cluster(self, conv_layers, nums_clusters_layers):

		mse_list = []

		for l, (nums_clusters, layer, sparsity_p) in enumerate(zip(nums_clusters_layers, conv_layers, self.sparsity_p)):

			if sparsity_p != 0:

				layer.weight.data, mse = self.cluster_filters_sparsity(layer.weight.data.cpu().numpy().astype('float32'),
									 nums_clusters, ratio=sparsity_p, seed=self.seed + l)
			else:

				layer.weight.data, mse = self.cluster_filters(layer.weight.data.cpu().numpy().astype('float32'),
									nums_clusters, seed=self.seed + l)

			mse_list.append(mse)

		return mse_list

	def initialize_data(self):

		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
											   download=True, transform=transform_test)
		self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.test_batch_size,
													  shuffle=False, num_workers=self.workers)

	def initialize_model(self):

		self.model = Wide_ResNet(depth=16, widen_factor=4, dropout_rate=0.3, num_classes=10).cuda()
		self.criterion = nn.CrossEntropyLoss().cuda()

	def load_retrained_model(self):

		model_file = os.path.join('./model', 'model_retrained_{}x.pth.tar'.format(self.retrained_model_CR))
		if os.path.isfile(model_file):
			checkpoint = torch.load(model_file)
			self.model.load_state_dict(checkpoint['state_dict'])
			print("=> Loaded checkpoint at '{}'".format(model_file))
		else:
			print("=> No checkpoint found at '{}'".format(model_file))

	def load_pretrained_model(self):

		model_file = os.path.join('./model', 'model_pretrained.pth.tar')
		if os.path.isfile(model_file):
			checkpoint = torch.load(model_file)
			self.model.load_state_dict(checkpoint['state_dict'])
			print("=> Loaded checkpoint at '{}'".format(model_file))
		else:
			print("=> No checkpoint found at '{}'".format(model_file))

	def initialize_all(self):

		self.initialize_data()
		self.initialize_model()
		print('-------------------------------------------- Wide ResNet (d=16 k=4) on CIFAR 10 --------------------------------------------')

	def get_parameters(self):

		self.samples_num_list = self.get_samples_num_list(self.conv_layers)

		print()
		print('---------------------------------------------------- Parameters Settings ---------------------------------------------------')
		print('Layer-wise N Samples:   {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<6} {:<6} {:<6} {:<6}'.format(*self.samples_num_list))
		print('Layer-wise K Clusters:  {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<6} {:<6} {:<6} {:<6}'.format(*self.clusters_num_list))
		print('Layer-wise Sparsity p:  {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<6} {:<6} {:<6} {:<6}'.format(*self.sparsity_p))

		print('----------------------------------------------------------------------------------------------------------------------------')
		print()

	def deploy(self):

		self.initialize_all()
		self.seek_layers(self.model)
		self.get_parameters()

		self.load_pretrained_model()
		self.seek_layers(self.model)
		self.get_memory()
		loss, prec = self.test(self.model)

		if self.visualize_filters:
			self.plot_conv(self.conv_layers, layer_indexes=self.visualize_layers_index, save_name='Pre-Trained Model')

		print()
		print('------------------------------------------------------------- Deep K-Means w/o Re-Training ----------------------------------------------------------')
		print('Pre-Trained Model: Accuracy: {:.2f}%, Loss: {:.4f}, Parameters: {:.2f} MB'.format(prec, loss, self.all_memory/(8*1024*1024)))

		print('-----------------------------------------------------------------------------------------------------------------------------------------------------')
		print('Layer-wise Memory Consumption {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}%'.format(
				*self.memory_percent_layer))

		print('-----------------------------------------------------------------------------------------------------------------------------------------------------')
		print()

		self.load_retrained_model()
		self.seek_layers(self.model)
		self.get_memory()
		loss, prec = self.test(self.model)
		if self.visualize_filters:
			self.plot_conv(self.conv_layers, layer_indexes=self.visualize_layers_index, save_name='Deep k-Means Re-Trained Model (Before Comp.)')

		print()
		print('------------------------------------------------------------ Deep K-Means w/ Re-Training ------------------------------------------------------------')
		print('Deep k-Means Re-Trained Model (Before Comp.): Accuracy: {:.2f}%, Loss: {:.4f}, Parameters: {:.2f} MB'.format(prec, loss, self.all_memory/(8*1024*1024)))

		mse_list = self.cluster(self.conv_layers, self.clusters_num_list)
		loss, prec = self.test(self.model)

		self.get_compressed_memory()

		print('Deep k-Means Re-Trained Model (After Comp.):  Accuracy: {:.2f}%, Loss: {:.4f}, Parameters: {:.2f} MB, Compression Ratio: {:.3f}, MMSE: {:.3e}'.format(
			   prec, loss, self.all_memory_compressed/(8*1024*1024), self.compression_ratio, np.mean(mse_list)))
		print('----------------------------------------------------------------------------------------------------------------------------------------------------------------')
		print('Layer-wise Memory Consumption (Before Comp.) {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}%'.format(
				*self.memory_percent_layer))
		print('Layer-wise Memory Consumption (After Comp.)  {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}% {:5.2f}%'.format(
				*self.compressed_memory_percent_layer))
		print('----------------------------------------------------------------------------------------------------------------------------------------------------------------')
		print()

		if self.visualize_filters:
			self.plot_conv(self.conv_layers, layer_indexes=self.visualize_layers_index, save_name='Deep k-Means Re-Trained Model (After Comp.)')

	def test(self, model):

		model.eval()
		test_loss = 0
		correct = 0
		for data, target in self.testloader:
			data, target = Variable(data.cuda()), Variable(target.cuda())
			output = model(data)
			test_loss += self.criterion(output, target).data[0]
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).cpu().sum()

		test_loss_avg = (test_loss * self.test_batch_size)/ len(self.testloader.dataset)
		acc = 100. * correct / len(self.testloader.dataset)
		return test_loss_avg, acc

	def plot_filters(self, weights, save_name):

		def add_subplot_border(ax, width=1, color=None):

			fig = ax.get_figure()

			x0, y0 = ax.transAxes.transform((0, 0))
			x1, y1 = ax.transAxes.transform((1, 1))

			x0, y0 = ax.transAxes.inverted().transform((x0, y0))
			x1, y1 = ax.transAxes.inverted().transform((x1, y1))

			rect = plt.Rectangle(
				(x0, y0), x1 - x0, y1 - y0,
				color=color,
				transform=ax.transAxes,
				zorder=-1,
				lw=2 * width + 1,
				fill=None,
			)
			fig.patches.append(rect)

		num_filters, channel, width, _ = weights.shape
		if not len(weights.shape) == 4:
			raise Exception("assumes a 4D weight")

		num_cols = int(np.round(np.sqrt(num_filters)))
		num_rows = int(np.ceil(num_filters / num_cols))

		num_sub_cols = int(np.round(np.sqrt(channel)))
		num_sub_rows = int(np.ceil(channel / num_sub_cols))

		fig = plt.figure(figsize=(num_cols*num_sub_cols, num_rows*num_sub_rows))

		outer_grid = gridspec.GridSpec(num_cols, num_cols, wspace=0.1, hspace=0.1)

		for i in range(num_cols*num_cols):

			inner_grid = gridspec.GridSpecFromSubplotSpec(num_sub_cols, num_sub_rows, subplot_spec=outer_grid[i], wspace=0.1, hspace=0.1)

			for j in range(num_sub_cols*num_sub_rows):

				ax = plt.Subplot(fig, inner_grid[j])
				add_subplot_border(ax, width=4, color='black')
				ax.imshow(weights[i, j, :, :])
				ax.set_xticks([])
				ax.set_yticks([])
				fig.add_subplot(ax)

		plt.show()
		fig.savefig(save_name, dpi=200, bbox_inches='tight')

	def plot_conv(self, conv_layers, layer_indexes, save_name):

		for layer_index in layer_indexes:

			savename = os.path.join('./visuals', 'Conv{} {}.png'.format(layer_index+1, save_name))

			self.plot_filters(weights=conv_layers[layer_index].weight.data.cpu().numpy(), save_name=savename)

		print("=> Saved filter visualization result under '{}'".format(savename))

if __name__ == '__main__':

	Compression_Rate = 45 # Compression_Rate = 45, 47, 50

	retrained_model_CR = Compression_Rate

	if Compression_Rate == 45: # Parameter Settings for Compression Rate = 45

		clusters_num_list = [144,
							  90, 80, 50, 30, 30,
							  100, 50, 10, 50, 50,
							  30, 7, 12, 7, 7]
		sparsity_p = [0,
					   0.3, 0.4, 0.5, 0.4, 0.4,
					   0.5, 0.5, 0.5, 0.5, 0.5,
					   0.62, 0.9, 0.5, 0.75, 0.9]

	elif Compression_Rate == 47: # Parameter Settings for Compression Rate = 47

		clusters_num_list = [144,
							 90, 80, 50, 30, 30,
							 100, 50, 10, 50, 50,
							 7, 6, 12, 7, 7]
		sparsity_p = [0,
					  0.3, 0.4, 0.5, 0.4, 0.4,
					  0.5, 0.5, 0.5, 0.5, 0.5,
					  0.6, 0.9, 0.5, 0.75, 0.9]

	elif Compression_Rate == 50: # Parameter Settings for Compression Rate = 50

		clusters_num_list = [144,
							 90, 80, 50, 30, 30,
							 100, 50, 10, 50, 50,
							 7, 6, 12, 7, 7]
		sparsity_p = [0,
					  0.3, 0.4, 0.5, 0.4, 0.4,
					  0.5, 0.5, 0.5, 0.5, 0.5,
					  0.87, 0.9, 0.5, 0.75, 0.9]

	visualize_filters = True

	visualize_layers_index = [1]

	deployer = deployer(retrained_model_CR=retrained_model_CR,
						clusters_num_list=clusters_num_list,
						sparsity_p=sparsity_p,
						visualize_filters=visualize_filters,
						visualize_layers_index=visualize_layers_index)

	deployer.deploy()
