
import os, argparse, time, copy, cv2, wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataloader import default_collate
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from pyexpat import model
from datetime import timedelta
from general_utils import load_cfg, get_cfg_from_full_path, get_datetime, bcolors
from general_utils import Print_Verbose
from launch_ruler import launch

def imshow(inp, title=None):
    
    inp = inp.cpu() if torch.device else inp
    inp = inp.numpy().transpose((1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(.005)

def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()
'''
# not a great visualization
def visualize_model(opts, net, num_images=4):
    images_so_far = 0
    fig = plt.figure(figsize=(15, 10))
    
    for i, data in enumerate(test_dataloader):
        inputs, labels = data
        if opts.use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        outputs = net(inputs)
        _, preds = torch.max(outputs.data, 1)
        preds = preds.cpu().numpy() if opts.use_cuda else preds.numpy()
        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(2, num_images//2, images_so_far)
            ax.axis('off')
            ax.set_title('predictes: {}'.format(test_dataset.classes[preds[j]]))
            imshow(inputs[j])
            
            if images_so_far == num_images:
                return 
'''
@dataclass
class TrainOptions:
    img_size: int = 720
    use_cuda: bool = True
    batch_size: int = 18
    learning_rate: float = 1e-3
    n_epochs: int  = 20
    print_every: int  = 5
    num_classes: int  = 2
    split_train_dir_automatically: bool = True
    split_train_dir_consistently: bool = True
    seed_for_random_split: int = 2022
    save_all_checkpoints: bool = False
    n_gpus: int = 1 
    n_machines: int = 1 
    default_timeout_minutes: int = 30

    dir_val: str = None

    path_to_config: str = field(init=False)
    path_to_model: str = field(init=False)
    path_to_ruler_class_names: str = field(init=False)

    dir_train: str = field(init=False)
    model_name: str = field(init=False)
    new_time: str = field(init=False)
    
    cfg: str = field(init=False)

    w_and_b_key: str = field(init=False)
    project: str = field(init=False)
    entity: str = field(init=False)

    def __post_init__(self) -> None:
        '''
        Setup
        '''
        self.new_time = get_datetime()
        self.default_timeout_minutes = timedelta(self.default_timeout_minutes)
        '''
        Configure names
        '''
        self.dir_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        path_cfg_private = os.path.join(self.dir_root,'PRIVATE_DATA.yaml')
        self.cfg_private = get_cfg_from_full_path(path_cfg_private)

        self.path_to_config = self.dir_root
        self.cfg = load_cfg(self.path_to_config)

        self.path_to_model = os.path.join(self.path_to_config, 'leafmachine2','machine','ruler_classifier','model')
        self.path_to_ruler_class_names = os.path.join('ruler_classifier','ruler_classes.txt')
        
        '''
        Weights and Biases Info
        https://wandb.ai/site
        '''
        if self.cfg_private['w_and_b']['w_and_b_key'] is not None:
            self.w_and_b_key = self.cfg_private['w_and_b']['w_and_b_key']
        if self.cfg_private['w_and_b']['ruler_classifier_project'] is not None:
            self.project = self.cfg_private['w_and_b']['ruler_classifier_project']
        if self.cfg_private['w_and_b']['entity'] is not None:
            self.entity = self.cfg_private['w_and_b']['entity']
        
        if self.cfg['leafmachine']['ruler_binarization_train']['dir_train'] is not None:
            self.dir_train = self.cfg['leafmachine']['ruler_binarization_train']['dir_train']
        else: 
            Print_Verbose(self.cfg,1,'ERROR: Training directory is missing').warning()

        if self.cfg['leafmachine']['ruler_binarization_train']['dir_val'] is not None:
            self.dir_val = self.cfg['leafmachine']['ruler_binarization_train']['dir_val']

        if self.cfg['leafmachine']['ruler_binarization_train']['split_train_dir_automatically'] is not None:
            self.split_train_dir_automatically = self.cfg['leafmachine']['ruler_binarization_train']['split_train_dir_automatically']

        if self.cfg['leafmachine']['ruler_binarization_train']['split_train_dir_consistently'] is not None:
            self.split_train_dir_consistently = self.cfg['leafmachine']['ruler_binarization_train']['split_train_dir_consistently']

        if self.cfg['leafmachine']['ruler_binarization_train']['save_all_checkpoints'] is not None:
            self.save_all_checkpoints = self.cfg['leafmachine']['ruler_binarization_train']['save_all_checkpoints']

        if self.cfg['leafmachine']['ruler_binarization_train']['n_machines'] is not None:
            self.n_machines = int(self.cfg['leafmachine']['ruler_binarization_train']['n_machines'])
        
        if self.cfg['leafmachine']['ruler_binarization_train']['n_gpus'] is not None:
            self.n_gpus = int(self.cfg['leafmachine']['ruler_binarization_train']['n_gpus'])

        if self.cfg['leafmachine']['ruler_binarization_train']['default_timeout_minutes'] is not None:
            self.default_timeout_minutes = int(self.cfg['leafmachine']['ruler_binarization_train']['default_timeout_minutes'])
        
        if self.cfg['leafmachine']['ruler_binarization_train']['img_size'] is not None:
            self.img_size = int(self.cfg['leafmachine']['ruler_binarization_train']['img_size'])

        if self.cfg['leafmachine']['ruler_binarization_train']['use_cuda'] is not None:
            self.use_cuda = self.cfg['leafmachine']['ruler_binarization_train']['use_cuda']

        if self.cfg['leafmachine']['ruler_binarization_train']['batch_size'] is not None:
            self.batch_size = int(self.cfg['leafmachine']['ruler_binarization_train']['batch_size'])

        if self.cfg['leafmachine']['ruler_binarization_train']['learning_rate'] is not None:
            self.learning_rate = float(self.cfg['leafmachine']['ruler_binarization_train']['learning_rate'])
            
        if self.cfg['leafmachine']['ruler_binarization_train']['n_epochs'] is not None:
            self.n_epochs = int(self.cfg['leafmachine']['ruler_binarization_train']['n_epochs'])

        if self.cfg['leafmachine']['ruler_binarization_train']['print_every'] is not None:
            self.print_every = int(self.cfg['leafmachine']['ruler_binarization_train']['print_every'])

        if self.cfg['leafmachine']['ruler_binarization_train']['num_classes'] is not None:
            self.num_classes = int(self.cfg['leafmachine']['ruler_binarization_train']['num_classes'])

        if self.cfg['leafmachine']['ruler_binarization_train']['seed_for_random_split'] is not None:
            self.num_classes = int(self.cfg['leafmachine']['ruler_binarization_train']['seed_for_random_split'])

        # If  model name is given, use it, else use default plus current time
        if self.cfg['leafmachine']['ruler_binarization_train']['model_name'] is not None:
            if not self.cfg['leafmachine']['ruler_binarization_train']['do_overwrite_model']:
                if self.cfg['leafmachine']['ruler_binarization_train']['model_name'] in os.listdir(self.path_to_model):
                    self.model_name = "__".join([self.cfg['leafmachine']['ruler_binarization_train']['model_name'], self.new_time])
                else:
                    self.model_name = self.cfg['leafmachine']['ruler_binarization_train']['model_name']
            else:
                self.model_name = self.cfg['leafmachine']['ruler_binarization_train']['model_name']
        else:
            self.model_name = "__".join(["ruler_classifier", self.new_time])

        # If dir_val is None assert split_train_dir_automatically = True
        if self.dir_val is None:
            self.split_train_dir_automatically = True


def train(opts):
    # Setup W & B
    wandb.login(key=opts.w_and_b_key)
    wandb.init(project=opts.project,name=opts.model_name.split('.')[0], entity=opts.entity, sync_tensorboard=False)

    # Define transformers, can add data augmentation here
    transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=[-0.4,0.4]),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.RandomGrayscale(p=0.1),
        torchvision.transforms.RandomRotation(45, resample=False,expand=False, center=None),
        torchvision.transforms.RandomAffine(45, translate=None, scale=None,shear=None, resample=False, fillcolor=0),
    ])
    scripted_transforms = transforms


    if opts.use_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    


    train_dataset = datasets.ImageFolder(root=opts.dir_train, transform=scripted_transforms)
    if opts.dir_val is not None:
        test_dataset = datasets.ImageFolder(root=opts.dir_val, transform=scripted_transforms)

    if opts.split_train_dir_automatically or opts.dir_val is None:
        tr = int(np.multiply(len(train_dataset.imgs),0.9))
        vr = len(train_dataset.imgs) - tr
        if opts.split_train_dir_consistently:
            train_dataset, test_dataset = random_split(train_dataset, [tr, vr], generator=torch.Generator().manual_seed(opts.seed_for_random_split))
        else:
            train_dataset, test_dataset = random_split(train_dataset, [tr, vr], generator=torch.Generator())

    train_dataloader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    test_dataloader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=True, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

    # images, labels = next(iter(train_dataloader)) 
    # print("images-size:", images.shape)
    # out = torchvision.utils.make_grid(images)
    # print("out-size:", out.shape)
    # imshow(out, title=[train_dataset.classes[x] for x in labels])
    '''
    net = models.resnet18(pretrained=True)
    # net = models.densenet161(pretrained=True)
    net = net.cuda().to(device) if device else net
    net
    '''
    if (torch.cuda.device_count() > 1) and (opts.n_gpus > 1):
        message = "".join(["Using ",str(torch.cuda.device_count()), " GPUs!"])
        print(f"{bcolors.OKGREEN}{message}{bcolors.ENDC}")
    elif (torch.cuda.device_count() == 1) and (opts.n_gpus == 1):
        message = "".join(["Using ",str(torch.cuda.device_count()), " GPU!"])
        print(f"{bcolors.OKGREEN}{message}{bcolors.ENDC}")
    else:
        message = "".join(["Using ",str(torch.cuda.device_count()), " CPU! Warning - very slow"])
        print(f"{bcolors.WARNING}{message}{bcolors.ENDC}")
    
    # net = models.resnext50_32x4d(pretrained=True)
    net = models.resnet18(pretrained=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=opts.learning_rate, momentum=0.9)

    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, opts.num_classes)
    # net.fc = net.fc.cuda() if opts.use_cuda else net.fc
    net.fc = net.fc.cuda() if opts.use_cuda else net.fc
    net.cuda()
    

    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    total_step = len(train_dataloader)
    for epoch in range(1, opts.n_epochs+1):
        running_loss = 0.0
        correct = 0
        total=0
        print(f'Epoch {epoch}\n')
        for batch_idx, (data_, target_) in enumerate(train_dataloader):
            data_, target_ = data_.cuda(), target_.cuda()
            optimizer.zero_grad()
            
            outputs = net(data_)
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _,pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred==target_).item()
            total += target_.size(0)
            if (batch_idx) % opts.print_every == 0:
                wandb.log({'batch-index': batch_idx, 'loss': loss.item()})
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch, opts.n_epochs, batch_idx, total_step, loss.item()))
        train_acc.append(100 * correct / total)
        _ta = round(100 * correct / total,4)
        train_loss.append(running_loss/total_step)
        _tl = round(np.mean(running_loss/total_step),4)
        print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
        wandb.log({'training-accuracy': _ta, 'training-loss': _tl})
        batch_loss = 0
        total_t=0
        correct_t=0
        with torch.no_grad():
            net.eval()
            for data_t, target_t in (test_dataloader):
                data_t, target_t = data_t.to(device), target_t.to(device)
                outputs_t = net(data_t)
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()
                _,pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t==target_t).item()
                total_t += target_t.size(0)
            val_acc.append(100 * correct_t/total_t)
            val_loss.append(batch_loss/len(test_dataloader))
            network_learned = batch_loss < valid_loss_min
            _va = round(100 * correct_t/total_t,4)
            _vl = round(np.mean(batch_loss/len(test_dataloader)),4)
            wandb.log({'validation-accuracy': _va, 'validation-loss': _vl})
            print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')

            
            if network_learned:
                valid_loss_min = batch_loss
                if opts.save_all_checkpoints:
                    print_loss = str(round(valid_loss_min,4)).replace(".","-")
                    ckpt_name = "".join(['last_checkpoint__Loss_',print_loss,'.pt'])
                    ckpt_name_s = "".join(['last_checkpoint_scripted__Loss_',print_loss,'.pt'])

                    torch.save(net.state_dict(), os.path.join(opts.path_to_model,ckpt_name))
                    model_scripted_ckpt = torch.jit.script(net) # Export to TorchScript
                    model_scripted_ckpt.save(os.path.join(opts.path_to_model,ckpt_name_s))
                    print('Improvement-Detected, save-model')
                else:
                    torch.save(net.state_dict(), os.path.join(opts.path_to_model,'last_checkpoint.pt'))
                    model_scripted_ckpt = torch.jit.script(net) # Export to TorchScript
                    model_scripted_ckpt.save(os.path.join(opts.path_to_model,'last_checkpoint_scripted.pt'))
                    print('Improvement-Detected, save-model')
        net.train()


    model_scripted = torch.jit.script(net) # Export to TorchScript
    model_scripted.save(os.path.join(opts.path_to_model,opts.model_name)) # Save
    print(f'train_acc = {train_acc}')
    print(f'train_acc = {val_acc}')

    fig = plt.figure(figsize=(20,10))
    plt.title("Train-Validation Accuracy")
    plt.plot(train_acc, label='train')
    plt.plot(val_acc, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.legend(loc='best')

    plt.ion()
    # visualize_model(net)
    plt.pause(5)
    plt.ioff()
    fig.savefig(os.path.join(opts.path_to_model,'.'.join([opts.model_name.split('.')[0], 'pdf'])),dpi=300, format='pdf')
    print(f'train_acc = {train_acc}')
    print(f'train_acc = {val_acc}')


# Step 2: Initialize the inference transforms
# preprocess = checkpoint.transforms()

# Step 3: Apply inference preprocessing transforms
# batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
# prediction = net(batch_t).squeeze(0).softmax(0)
# class_id = prediction.argmax().item()
# score = prediction[class_id].item()
# category_name = checkpoint.meta["categories"][class_id]
# print(f"{category_name}: {100 * score:.1f}%")


# out = net(batch_t)
# _, index = torch.max(out, 1)
# percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
# print(classes[index[0]], percentage[index[0]].item())
# _, indices = torch.sort(out, descending=True)
# [(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]



# with torch.no_grad():
#     out_data = net(img)


# netTR = torch.load()

# netTR.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# netTR.eval()
# netTR = netTR.cuda() if device else netTR
# # outputs = netTR(cv2.imread('E:/TEMP_ruler/Rulers_ByType_Squarify/block_alternate_cm/Ruler__Herbarium-of-Andalas-University_2609484175_Cannabaceae_Trema_cannabina__1.jpg'))
# # _, preds = torch.max(outputs.data, 1)
# # preds = preds.cpu().numpy() if use_cuda else preds.numpy()
# visualize_model(netTR)
# plt.ioff()

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    # load tarining opitions
    opts = TrainOptions()
    # args = parse_train_ruler_classifier()
    launch(
        train,
        opts.n_gpus,
        opts.n_machines,
        machine_rank=0,
        dist_url="auto",
        opts=opts,
        timeout=opts.default_timeout_minutes,
    )
    # train()
