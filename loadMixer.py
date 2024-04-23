import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# Assume your model's code is in a file named mlp_mixer_pytorch.py
from models.maskedmlpmixer import MLPMixer
#from models.mlpmixer import MLPMixer
import torchvision
import pdb
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


# Parameters
image_size = 32  # CIFAR-10 images are 32x32
channels = 3  # RGB channels
patch_size = 4  # Example patch size
dim = 512  # Dimension of the token/patch embeddings
depth = 6  # Number of blocks
num_classes = 10  # Number of classes in CIFAR-10
#model_weights_path = '/p/dataset/abhishek/mlpmixer0_1-4-ckpt.t7'  # Path to your model's weights
#model_weights_path = '/p/dataset/abhishek/mlpmixerrand0_25-4-ckpt.t7'
#model_weights_path =  '/p/dataset/abhishek/mlpmixerband8a0_25-4-ckpt.t7'
model_weights_path =  '/p/dataset/abhishek/mlpmixerbandSmall0_25-4-ckpt.t7'

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load CIFAR-10 test dataset
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

# Load model
model = MLPMixer(
    image_size=image_size,
    channels=channels,
    patch_size=patch_size,
    dim=dim,
    depth=depth,
    num_classes=num_classes,
)
# Load the entire checkpoint
checkpoint = torch.load(model_weights_path)

# Extract the model's state dictionary
model_state_dict = checkpoint['model']
#pdb.set_trace()

# Load the model's state dictionary
model.load_state_dict(model_state_dict)
#pdb.set_trace() 
####################################################### Model Structure #######################################################

# Get the size, type, and module name of all weight matrices in the model. Also get the number of non zeros in each weight matrix
""" total_params = 0
total_non_zeros = 0

for name, param in model.named_parameters():
    if 'weight' in name:
        module_name = name.split('.')[0]
        module_type = type(getattr(model, module_name)).__name__
        print(f'{name}: {param.size()} - {param.type()} - {module_type}')
        non_zeros = torch.count_nonzero(param)
        total_params += param.numel()
        total_non_zeros += non_zeros.item()
        print(f'Number of non zeros: {non_zeros.item()}')

non_zeros_percentage = (total_non_zeros / total_params) * 100
print(f'Total number of non zeros as a percentage of all parameters: {non_zeros_percentage}%')
 """

# Get the weights of layer 6.1.fn.0.weight
weights1 = model[1].weight.data
#weights2 = model[2][0].fn[0].weight.data[:, :, :-1]
#weights3 = model[2][0].fn[3].weight.data[:, :, :-1]
weights2 = model[2][0].fn[0].weight.data.squeeze()
weights3 = model[2][0].fn[3].weight.data.squeeze()
weights4 = model[2][1].fn[0].weight.data
weights5 = model[2][1].fn[3].weight.data
weights6 = model[10].weight.data

# Create a mask to identify the nonzero values
mask1 = weights1 != 0
mask2 = weights2 != 0
mask3 = weights3 != 0
mask4 = weights4 != 0
mask5 = weights5 != 0
mask6 = weights6 != 0

# Apply the mask to the weights
masked_weights1 = weights1[mask1]
masked_weights2 = weights2[mask2]
masked_weights3 = weights3[mask3]
masked_weights4 = weights4[mask4]
masked_weights5 = weights5[mask5]
masked_weights6 = weights6[mask6]

# Get the dimensions of the weights matrices
num_rows1, num_cols1 = weights1.size()
#num_rows2, num_cols2 = weights2.size()[:2] if len(weights2.size()) > 1 else (0, 0)  # Include an additional dimension for the mask
#num_rows3, num_cols3 = weights3.size()[:2] if len(weights3.size()) > 1 else (0, 0)  # Include an additional dimension for the mask
num_rows2, num_cols2 = weights2.size()
num_rows3, num_cols3 = weights3.size()
num_rows4, num_cols4 = weights4.size()
num_rows5, num_cols5 = weights5.size()
num_rows6, num_cols6 = weights6.size()

# Create matrices of the same dimensions as weights with all zeros
plot_matrix1 = torch.zeros(num_rows1, num_cols1)
plot_matrix2 = torch.zeros(num_rows2, num_cols2)
plot_matrix3 = torch.zeros(num_rows3, num_cols3)
plot_matrix4 = torch.zeros(num_rows4, num_cols4)
plot_matrix5 = torch.zeros(num_rows5, num_cols5)
plot_matrix6 = torch.zeros(num_rows6, num_cols6)


print("Weights2 size:", weights2.size())
print("Mask2 size:", mask2.size())
# Set the values of the plot matrices to 1 where the weights matrices have non-zero values
""" if len(weights2.size()) > 1:
    mask2_flat = mask2[:,:,0].flatten()  # Flatten the mask to 1D
    plot_matrix2[mask2_flat] = 1
else:
    plot_matrix2[mask2] = 1

if len(weights3.size()) > 1:
    mask3_flat = mask3[:,:,0].flatten()  # Flatten the mask to 1D
    plot_matrix3[mask3_flat] = 1
else:
    plot_matrix3[mask3] = 1
 """
plot_matrix1[mask1] = 1
plot_matrix2[mask2] = 1
plot_matrix3[mask3] = 1
plot_matrix4[mask4] = 1
plot_matrix5[mask5] = 1
plot_matrix6[mask6] = 1
# Plot the matrices
plt.imshow(plot_matrix1, cmap='gray')
plt.savefig('weights_plot1.png')

plt.imshow(plot_matrix2, cmap='gray')
plt.savefig('weights_plot2.png')

plt.imshow(plot_matrix3, cmap='gray')
plt.savefig('weights_plot3.png')

plt.imshow(plot_matrix4, cmap='gray')
plt.savefig('weights_plot4.png')

plt.imshow(plot_matrix5, cmap='gray')
plt.savefig('weights_plot5.png')

plt.imshow(plot_matrix6, cmap='gray')
plt.savefig('weights_plot6.png')

""" # Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set model to evaluation mode
pdb.set_trace()
# Evaluation
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the 10000 test images: {100 * correct / total}%')
 """