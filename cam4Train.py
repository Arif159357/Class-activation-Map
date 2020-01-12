import cv2
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))



num_epochs = 5
batch_size = 8
learning_rate = 0.1

img_transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

train_data_dir = datasets.ImageFolder("train2", transform = img_transform)

#print(len(train_data_dir))
train_size = int(0.8 * len(train_data_dir))
test_size = len(train_data_dir) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(train_data_dir, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__() 

        self.fc=nn.Linear(512,6, bias = False)
     
    def forward(self,x):
        
        dim = x.shape[0]
        v=x.view(dim,512,-1)
        x=v.mean(2)
        x=x.view(1,dim,512)
        x=self.fc(x)


        return  x.view(-1,6)

   
net = models.vgg16(pretrained=True) 
mod = nn.Sequential(*list(net.children())[:-1])

model=nn.Sequential(mod,Net())

print("complete")

trainable_parameters = []
for name, p in model.named_parameters():
    #print(name)
    if "fc" in name:
        trainable_parameters.append(p)
optimizer = torch.optim.SGD(params=trainable_parameters, lr=0.1, momentum=1e-5)  
criterion = nn.CrossEntropyLoss()


total_step = len(train_loader)
loss_list = []
acc_list = []

min_loss=9999
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Run the forward pass
    
        outputs = model(images)
       
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if (i % 100) == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))
    
    with open('loss.txt', 'w+') as f:
        f.write("%s\n" % loss)
    if loss < min_loss:
        min_loss = loss
        torch.save(model.state_dict(), 'training4.pth')
      
##########CAM running
#model.load_state_dict(torch.load('training.pth'))       
#model.eval()       
#features_blobs = []
#def hook_feature(module, input, output):
#    features_blobs.append(output.data.numpy())
#
#mod.register_forward_hook(hook_feature)
#
#print(features_blobs)    
#params = list(Net().parameters())
#weight_softmax = np.squeeze(params[-2].data.numpy())    
#
#def returnCAM(feature_conv, weight_softmax, class_idx):
#    # generate the class -activation maps upsample to 256x256
#    size_upsample = (256, 256)
#    bz, nc, h, w = feature_conv.shape
#    #print(feature_conv.shape)
#    output_cam = []
#    for idx in class_idx:
##        print(weight_softmax[idx])
#        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
##        print(cam.shape)
##        cam = cam.view(49,7)
##        print(cam.shape)
#        cam = cam.reshape(h, w)
#        cam = cam - np.min(cam)
#        cam_img = cam / np.max(cam)
#        cam_img = np.uint8(255 * cam_img)
#        output_cam.append(cv2.resize(cam_img, size_upsample))
#    return output_cam
#
#normalize = transforms.Normalize(
#   mean=[0.485, 0.456, 0.406],
#   std=[0.229, 0.224, 0.225]
#)
#preprocess = transforms.Compose([
#   transforms.Resize((224,224)),
#   transforms.ToTensor(),
#   normalize
#])
#list1 = open('test1.txt','r')
#IMG_URL = list1.readlines()
#org_loc ='test1/test/'    

#for fname in IMG_URL:
#    
#    fname = fname.rstrip('\n')    
#    img_pil = Image.open(org_loc+fname+'.png')
#    #img_pil.save('test.jpg')    -
#
#    img_tensor = preprocess(img_pil)
# 
#    img_variable = Variable(img_tensor.unsqueeze(0))
#   
#    logit = model(img_variable)
# 
#
## download the imagenet category list
##    classes = {int(key):value for (key, value)
##              in requests.get(LABELS_URL).json().items()}
#
#    h_x = F.softmax(logit, dim=1).data.squeeze()
# 
#    probs, idx = h_x.sort(0, True)
#    probs = probs.detach().numpy()
#    idx = idx.numpy()
#    print(idx)
## output the prediction
##    for i in range(0, 5):
##        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))-
#
## generate class activation mapping for the top1 prediction
#    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
#
## render the CAM and output
#   
#        
##    print('output for the top1 prediction: %s'%classes[idx[0]])
#    img = cv2.imread(org_loc+fname+'.png')
#    
#    height, width, _ = img.shape
#    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
#    result = heatmap * 0.3 + img * 0.5
#    path = 'CAM_Fire/'
#    cv2.imwrite('test2/'+fname+'.png', result)        