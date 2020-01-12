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
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

num_epochs = 10
batch_size = 8
learning_rate = 0.1

img_transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

train_data_dir = datasets.ImageFolder("train2", transform = img_transform)

#print(len(train_data_dir))
#train_size = int(0.8 * len(train_data_dir))
#test_size = len(train_data_dir) - train_size
#train_dataset, test_dataset = torch.utils.data.random_split(train_data_dir, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(
train_data_dir, batch_size=batch_size, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__() 

        self.fc=nn.Linear(512,6, bias = False)
     
    def forward(self,x):
        
        dim = x.shape[0]
        v=x.view(dim,512,-1)
        x=v.mean(2)
        x=x.view(1,dim,512)
        x= F.sigmoid(self.fc(x))


        return  x.view(-1,6)

   
net = models.vgg16(pretrained=True) 
mod = nn.Sequential(*list(net.children())[:-1])

model=nn.Sequential(mod,Net())

print("complete")


model.load_state_dict(torch.load('training2.pth',map_location='cpu'))       
model.eval()       

#def hook_feature(module, input, output):
#         features_blobs.append(output.data.cpu().numpy())
#print(features_blobs)    
params = list(Net().parameters())
weight_softmax = np.squeeze(params[-1].data.numpy())  
 

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class -activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    print(feature_conv.shape)
    output_cam = []
    for idx in class_idx:
#        print(weight_softmax[idx])
        beforeDot =  feature_conv.reshape((nc, h*w))
        cam = np.matmul(weight_softmax[idx], beforeDot)
#        print(cam.shape)
#        cam = cam.view(49,7)
#        print(cam.shape)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])
list1 = open('testImages.txt','r')
IMG_URL = list1.readlines()
org_loc ='test/test/'    


res = list(map(lambda i: i[ : -5], IMG_URL)) 
print(res)

test_labels = []
for i in res:
    if i =='firesdisaster':
        test_labels.append(0)
        
    if i =='flooddisaster':
        test_labels.append(1)
        
    if i == 'humandisaster':
        test_labels.append(2)
    
    if i == 'infrastructure':
        test_labels.append(3)    

    if i == 'naturedisaster':
        test_labels.append(4)
    
    if i == 'nondisaster':
        test_labels.append(5)    


predicted_labels = []
for fname in IMG_URL:
    
    fname = fname.rstrip('\n')    
    img_pil = Image.open(org_loc+fname+'.png')
    #img_pil.save('test.jpg')    -
    
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    logit = model(img_variable)
 

# download the imagenet category list
#    classes = {int(key):value for (key, value)
#              in requests.get(LABELS_URL).json().items()}

    h_x = F.softmax(logit, dim=1).data.squeeze()
 
    probs, idx = h_x.sort(0, True)
    probs = probs.detach().numpy()
    idx = idx.numpy()
    
    predicted_labels.append(idx[0])
    predicted =  train_loader.dataset.classes[idx[0]]
    
    print("Target: " + fname + " | Predicted: " +  predicted)
   
    # output the prediction
    #for i in range(0, 5):
    #   print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))-

# generate class activation mapping for the top1 prediction
    
    logitModel = logit.cpu().detach().numpy()

    features_blobs = mod(img_variable)
    features_blobs1 = features_blobs.cpu().detach().numpy()
    features_blobs1_avgpool = features_blobs.view(512,7*7).mean(1).view(1,-1)
    features_blobs1_avgpool = features_blobs1_avgpool.cpu().detach().numpy()
    logitManual = np.matmul(features_blobs1_avgpool, weight_softmax.transpose())
    CAMs = returnCAM(features_blobs1, weight_softmax, [idx[0]])

# render the CAM and output
   
        
#    print('output for the top1 prediction: %s'%classes[idx[0]])
    readImg = org_loc+fname+'.png'
    img = cv2.imread(readImg)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.5 + img * 0.5
   
    saveImg = 'newtrainoldtest/'+fname +'_Predicted_' + str(predicted)  + '_prob_'+str(probs[0]) + '_.png'
    cv2.imwrite(saveImg, result)


print(predicted_labels)
cm = confusion_matrix(test_labels,predicted_labels)
print(cm)