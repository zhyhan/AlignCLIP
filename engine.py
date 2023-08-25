import time
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from CLIP.clip import clip
#from CLIP.clip import model as cmodel
import torch.nn as nn
from CLIP.clip import model_surgery as cmodel
import converter_dassl, converter_domainbed
from utils import accuracy, AverageMeter, ProgressMeter, TensorboardWriter, ForeverDataIterator
from  matplotlib import pyplot as plt
import seaborn as sns



class GeneralMovingAverage(object):
    def __init__(self, model, weight_func):
        self.model = model
        self.weight_func = weight_func
        self.iter = 0
        self.weight = weight_func(self.iter)
        self.weight_sum = self.weight
        self.moving_avg = copy.deepcopy(model)
        for param in self.moving_avg.parameters():
            param.requires_grad = False

    def update(self):
        self.iter += 1
        self.weight = self.weight_func(self.iter)
        relative_weight = self.weight / self.weight_sum
        for moving_avg_param, param in zip(self.moving_avg.parameters(), self.model.parameters()):
            moving_avg_param.data = (moving_avg_param + relative_weight * param) / (1 + relative_weight)
        self.weight_sum += self.weight

    def __call__(self, x: torch.Tensor):
        return self.moving_avg(x)

    def train(self, mode=True):
        self.moving_avg.train(mode)

    def eval(self):
        self.train(False)

    def state_dict(self):
        return self.moving_avg.state_dict()

    def load_state_dict(self, state_dict):
        self.moving_avg.load_state_dict(state_dict)

    @property
    def module(self):
        return self.moving_avg.module


def get_dataset(args):
    if args.task == "domain_shift":
        # load domainbed data
        train_datasets, val_datasets, test_datasets, class_names = \
            converter_domainbed.get_domainbed_datasets(dataset_name=args.data, root=args.root, targets=args.targets, holdout=0.2)
        train_class_names = class_names
        train_iter = converter_domainbed.get_forever_iter(train_datasets, args.batch_size, num_workers=args.workers)
        val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        test_loaders = [
            {
                "name": "test",
                "loader": DataLoader(ConcatDataset(test_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                "class_names": class_names
            }
        ]
        template = "a photo of a {}."
    
    elif args.task == "open_class":
        # load dassl data
        train_dataset, val_dataset, test_dataset, open_dataset, base_class_names, open_class_names, template = \
            converter_dassl.get_dassl_datasets(dataset_name=args.data, root=args.root, n_shot=args.n_shot)
        train_class_names = base_class_names
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
        train_iter = ForeverDataIterator(train_loader)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        test_loaders = [
            {
                "name": "test",
                "loader": DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                "class_names": base_class_names
            },
            {
                "name": "open",
                "loader": DataLoader(open_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                "class_names": open_class_names
            }
        ]

    elif args.task == "in_the_wild":
        # load domainbed data
        train_datasets, val_datasets, test_datasets, open_datasets, base_class_names, open_class_names = \
            converter_domainbed.get_domainbed_datasets(dataset_name=args.data, root=args.root, targets=args.targets, holdout=0.2, open_ratio=0.5)
        train_class_names = base_class_names
        train_iter = converter_domainbed.get_forever_iter(train_datasets, args.batch_size, num_workers=args.workers)
        val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        test_loaders = [
            {
                "name": "test",
                "loader": DataLoader(ConcatDataset(test_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                "class_names": base_class_names + open_class_names
            },
            {
                "name": "open",
                "loader": DataLoader(ConcatDataset(open_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                "class_names": base_class_names + open_class_names
            }
        ]
        template = "a photo of a {}."
    
    return train_iter, val_loader, test_loaders, train_class_names, template


def get_text_features(clip_model, template, class_names, device, args):
    with torch.no_grad():
        texts = torch.cat(
            [clip.tokenize(template.format(c.replace("_", " ")))
            for c in class_names]).to(device)
        text_features = clip_model.encode_text(texts)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        #matrix
        similarity_matrix = text_features @ text_features.T
        # min_val = torch.min(similarity_matrix)
        # max_val = torch.max(similarity_matrix)
        # similarity_matrix = (similarity_matrix - min_val) / (max_val - min_val)

        # distribution = similarity_matrix / torch.sum(similarity_matrix, dim=1, keepdim=True)
        distribution = F.softmax(similarity_matrix*args.smoothing, dim=1)
        class_num = len(similarity_matrix)
        #save similarity_matrix
        sim_array = distribution.cpu().numpy()
        plt.figure(figsize=(20,20))
        # Using seaborn heatmap
        ax = sns.heatmap(sim_array, cmap='coolwarm', annot=True, yticklabels=class_names, xticklabels=class_names, annot_kws={"size": 5})
        ax.xaxis.tick_top() # x axis on top
        ax.xaxis.set_label_position('top')
        plt.xticks(rotation=90, va="center", ha='right')
        plt.title('Similarity Matrix', y=1.06)
        plt.savefig('figs/sm_{}.png'.format(class_num), dpi=600, bbox_inches='tight')
        #plt.show()
    return text_features, distribution


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


def train(train_iter: ForeverDataIterator, model, moving_avg_model: GeneralMovingAverage, text_features: torch.Tensor,
          optimizer, lr_scheduler, epoch: int, args, writer: TensorboardWriter, device, smooth_loss):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    alosses = AverageMeter('aloss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, alosses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # Freeze all Norm Layers
    model.eval()

    end = time.time()
    for i in range(args.iters_per_epoch):
        # obtain data
        x = []
        labels = []
        if args.task in ["domain_shift", "in_the_wild"]:
            for x_d, labels_d in next(train_iter):
                x.append(x_d)
                labels.append(labels_d)
            x, labels = torch.cat(x), torch.cat(labels)
        else:
            x, labels = next(train_iter)
        x, labels = x.to(device), labels.to(device)

        # measure data loading time
        data_time_step = time.time() - end
        data_time.update(data_time_step)

        # compute output
        f, f_s, attn_output_weights_list = model(x.half()) #f is raw prediction, f_s is with surgery
        f = f / f.norm(dim=-1, keepdim=True)
        #compute similarity
        f[:,0,:] -= args.alpha * text_features[labels]
        similarity_f = f @ text_features.T
        y = similarity_f[:,0,:]
        y = args.temperature * y

        loss = smooth_loss(y, labels)
        #loss = F.cross_entropy(y, labels)

        f_s = f_s / f_s.norm(dim=-1, keepdim=True)
        #compute similarity
        similarity_fs = f_s[:,1:,:] @ text_features.T

        aloss = attention_align_loss(attn_output_weights_list, similarity_fs, labels, args)
        aloss_last = similarity_align_loss(similarity_f[:,1:,:], similarity_fs, labels)
        #total_loss = loss + args.aloss * (aloss)
        total_loss = loss + args.aloss * (aloss+aloss_last)
        cls_acc = accuracy(y, labels)[0]
        losses.update(loss.item(), x.size(0))
        alosses.update(aloss.item(), x.size(0))
        cls_accs.update(cls_acc.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        convert_models_to_fp32(model)
        optimizer.step()
        cmodel.convert_weights(model)
        lr_scheduler.step()

        moving_avg_model.update()

        bma_weight = moving_avg_model.weight

        # measure elapsed time
        batch_time_step = time.time() - end
        batch_time.update(batch_time_step)

        writer.record_training_values(
            {
                "Loss": (loss.item(), x.shape[0]),
                "Acc@1": (cls_acc.item(), x.shape[0]),
                "aloss": (aloss.item(), x.shape[0]),
                "Time": (batch_time_step,),
                "Data": (data_time_step,),
                "Weight": (bma_weight,),
            }
        )

        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def attention_align_loss(attn_output_weights_list, attention_distribution, label, args):
    class_token_attention = []
    for attn in attn_output_weights_list[:args.layer]:
        #take the class token attention
        # if args.layer > 6:
        #     attn = torch.mean(attn, dim=1)#multiple heads
        attn = F.normalize(attn[:, 0, 1:], p=1, dim=1)
        class_token_attention.append(attn)
    class_token_attention = torch.stack(class_token_attention)
    class_token_attention = torch.mean(class_token_attention, dim=0)
    #remove the class token attention

    # 2. 计算目标分布，即均匀分布或者相似度分布
    label = label.view(-1, 1, 1).expand(-1, 196, 1) # label: [108, 1, 1]
    similarity_fs_selected = attention_distribution.gather(2, label)  # similarity: [108, 196, 1]
    similarity_fs_selected = similarity_fs_selected.squeeze(2)  # similarity: [108, 196]
    similarity_fs = F.softmax(similarity_fs_selected, dim=1)

    kl_div = F.kl_div(class_token_attention.log(), similarity_fs, reduction='batchmean')
    #kl_div = torch.tensor(0.)
    return kl_div

class AdaptiveLabelSmoothingLoss(nn.Module):
    def __init__(self, distribution, smoothing=0.05):
        super(AdaptiveLabelSmoothingLoss, self).__init__()
        self.distribution = distribution
        #assert 0 <= smoothing < 1
        self.num_classes = self.distribution.size(1)
        self.smoothing = smoothing
        self.confidence = 0.9

    def forward(self, pred, target):
        #one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        #adaptive_one_hot = one_hot * self.confidence + self.distribution[target]
        #smooth_label = adaptive_one_hot + (1 - adaptive_one_hot) * self.smoothing / (self.num_classes - 1)
        #smooth_label = one_hot * self.confidence + (1 - one_hot) * self.smoothing / (self.num_classes*100.0 - 1)
        #adaptive_smooth_label = smooth_label + self.distribution[target]/(self.num_classes*100.0)

        #在类别数比较多的时候，有些GT class的confident小于0.7，因此需要重新调整一下。

        #standard label smoothing
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        adaptive_smooth_label = one_hot * self.confidence + (1 - one_hot) * self.distribution[target]

        #adaptaive label smoothing#在类别数比较多的时候，有些GT class的confident小于0.7，因此需要重新调整一下。
        #adaptive_smooth_label = self.distribution[target]
        #adaptive_smooth_label[target] == self.confidence
        loss = (-F.log_softmax(pred, dim=1) * adaptive_smooth_label).sum(dim=1).mean()
        return loss

def similarity_align_loss(similarity_f, similarity_fs, label):
    
    # First, you need to reshape your label tensor to match the dimensions of your similarity tensor.
    # The extra dimensions will have a size of 1 and will be automatically broadcasted to the necessary size.
    label = label.view(-1, 1, 1).expand(-1, similarity_f.size(1), 1) # label: [108, 1, 1] > [108, 196, 1]
    # Then, you can use the gather function. Note that the dimensions of label and similarity match.
    similarity_f_selected = similarity_f.gather(2, label)  # similarity: [108, 196, 1]
    similarity_fs_selected = similarity_fs.gather(2, label)  # similarity: [108, 196, 1]

    # Lastly, you can remove the extra dimension.
    similarity_f_selected = similarity_f_selected.squeeze(2)  # similarity: [108, 196]
    similarity_fs_selected = similarity_fs_selected.squeeze(2)  # similarity: [108, 196]
    similarity_f = F.softmax(similarity_f, dim=1)
    similarity_fs = F.softmax(similarity_fs, dim=1)

    kl_div = F.kl_div(similarity_f.log(), similarity_fs, reduction='batchmean')
    return kl_div

def validate(val_loader, model, text_features, args, device, shift=0) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.half().to(device)
            target = target.half().to(device) - shift

            # compute output
            image_features, _, _ = model(images)
            image_features = image_features[:,0,:] 
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # measure accuracy and record loss
            output_similarity = image_features @ text_features.T
            #output_similarity = output_similarity[:,0,:]
            acc1, = accuracy(output_similarity, target, topk=(1,))

            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    
    return top1.avg


def evaluate_all(model, val_loader, train_text_features, test_loaders, args, writer, device):
    print("Evaluate on validation set...")
    val_acc1 = validate(val_loader, model, train_text_features, args, device)
    writer.write_eval_values({"Acc@1": val_acc1}, prefix="val")

    for test_loader in test_loaders:
        split_name = test_loader["name"]
        print(f"Evaluate on {split_name} set...")
        validate(test_loader["loader"], model, test_loader["text_features"], args, device)
        writer.write_eval_values({"Acc@1": val_acc1}, prefix=split_name)

    return val_acc1