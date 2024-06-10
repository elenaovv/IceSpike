import torch
import os
import torch.nn as nn
import pickle
import argparse
import torch.nn.functional as F
import torch.optim as optim
from model import new_spikformer
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataset import TxtDataset, TxtDataset2
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer
from spikingjelly.activation_based import encoding
from spikingjelly.activation_based import functional
from utils.public import set_seed
from torchmetrics.classification import MatthewsCorrCoef

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,0,2,3"

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))


def to_device(x, device):
    for key in x:
        x[key] = x[key].to(device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dataset_name", default="igc_full", type=str)
    # parser.add_argument("--data_augment", default="True", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--fine_tune_lr", default=1e-2, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--teacher_model_path", default="", type=str)
    parser.add_argument("--label_num", default=4, type=int)
    parser.add_argument("--depths", default=6, type=int)
    parser.add_argument("--max_length", default=64, type=int)
    parser.add_argument("--dim", default=768, type=int)
    parser.add_argument("--ce_weight", default=0.0, type=float)
    parser.add_argument("--emb_weight", default=1.0, type=float)
    parser.add_argument("--logit_weight", default=1.0, type=float)
    parser.add_argument("--rep_weight", default=5.0, type=float)
    parser.add_argument("--num_step", default=32, type=int)
    parser.add_argument("--tau", default=10.0, type=float)
    parser.add_argument("--common_thr", default=1.0, type=float)
    parser.add_argument("--predistill_model_path", default="", type=str)
    # parser.add_argument("--predistill_requires_grad", default="True", type=str)
    parser.add_argument("--ignored_layers", default=1, type=int)
    # parser.add_argument("--metric", default="acc", type=str)
    parser.add_argument("--temperature", default=1.0, type=float)  # New parameter

    return parser.parse_args()


def distill(args):
    print(os.path.exists(args.teacher_model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained(args.teacher_model_path)
    teacher_model = RobertaForSequenceClassification.from_pretrained(args.teacher_model_path, num_labels=args.label_num,
                                                                     output_hidden_states=True).to(device)

    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()

    student_model = new_spikformer(depths=args.depths, length=args.max_length, T=args.num_step, \
                                   tau=args.tau, common_thr=args.common_thr, vocab_size=len(tokenizer), dim=args.dim,
                                   num_classes=args.label_num, mode="distill")

    # load embedding layer
    # student_model.emb.weight = teacher_model.roberta.embeddings.word_embeddings.weight
    # student_model.emb.weight.requires_grad = True

    if args.predistill_model_path != "":
        weights = torch.load(args.predistill_model_path)
        # for key in weights.keys():
        #     if "module.transforms" not in key:
        #         weights[key] = weights[key].float()
        #         if args.predistill_requires_grad == "False":
        #             weights[key].requires_grad = False
        #         elif args.predistill_requires_grad == "True":
        #             weights[key].requires_grad = True
        student_model.load_state_dict(weights, strict=False)
        print("load predistill model finish!")

    scaler = torch.cuda.amp.GradScaler()
    optimer = torch.optim.AdamW(params=student_model.parameters(), lr=args.fine_tune_lr, betas=(0.9, 0.999),
                                weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimer, T_max=args.epochs, eta_min=0)
    # scheduler_warmup = GradualWarmupScheduler(optimer, multiplier=1, total_epoch=5, after_scheduler=scheduler)

    # if args.data_augment == "True":
    #    print("With Augmentation")
    #    train_dataset = TxtDataset(data_path=f"data/{args.dataset_name}/train_augment.txt")
    # else:
    #    print("Without Augmentation")
    # train_dataset = TxtDataset(data_path=f"data/{args.dataset_name}/train.txt")
    train_dataset = TxtDataset2(data_path=f"/users/home/elenao23/2_finetuning_fix/data/{args.dataset_name}/train.txt")
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    test_dataset = TxtDataset2(data_path=f"/users/home/elenao23/2_finetuning_fix/data/{args.dataset_name}/test.txt")
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    valid_dataset = TxtDataset2(
        data_path=f"/users/home/elenao23/2_finetuning_fix/data/{args.dataset_name}/validation.txt")
    valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device_ids = [i for i in range(torch.cuda.device_count())]
    print(device_ids)
    if len(device_ids) > 1:
        student_model = nn.DataParallel(student_model, device_ids=device_ids).to(device)
    student_model = student_model.to(device)

    metric_list = []
    for epoch in tqdm(range(args.epochs)):
        # if epoch == 5:
        #     args.rep_weight == 0
        total_loss_list = []
        embeddings_loss_list = []
        ce_loss_list = []
        logit_loss_list = []
        rep_loss_list = []
        for batch in tqdm(train_data_loader):
            student_model.train()
            batch_size = len(batch[0])
            labels = batch[1].to(device)
            inputs = tokenizer(batch[0], padding="max_length", truncation=True, \
                               return_tensors="pt", max_length=args.max_length)
            # inputs = tokenizer(batch[0], padding=True, truncation=True, \
            #     return_tensors="pt", max_length=args.max_length)

            to_device(inputs, device)
            with torch.no_grad():
                teacher_outputs = teacher_model(**inputs)
            # [1:] means 12 outputs of each layer; [::x] means get value every x layers
            tea_embeddings = teacher_model.roberta.embeddings.word_embeddings.weight
            if len(device_ids) > 1:
                stu_embeddings = student_model.module.emb.weight
            else:
                stu_embeddings = student_model.emb.weight

            embeddings_loss = F.mse_loss(stu_embeddings, tea_embeddings)
            embeddings_loss_list.append(embeddings_loss.item())

            tea_rep = teacher_outputs.hidden_states[1:][::int(12 / args.depths)]  # layers output
            # len(stu_rep) = depth
            # stu_rep[0] shape: B L D
            stu_rep, student_outputs = student_model(inputs['input_ids'])

            student_outputs = student_outputs.reshape(-1, args.num_step, args.label_num)

            # Before transpose: B T Label_num
            # After transpose:  T B Label_num
            student_outputs = student_outputs.transpose(0, 1)

            student_logits = torch.mean(student_outputs, dim=0)  # B Label_num
            # last step
            # student_logits =  student_outputs[-1,:,:]# B Label_num

            # print("student_logits.shape: ", student_logits.shape)
            # print("labels.shape: ", labels.shape)
            ce_loss = F.cross_entropy(student_logits, labels)
            ce_loss_list.append(ce_loss.item())
            # print("ce_loss: ", ce_loss, ce_loss.dtype)

            # print("student_logits.shape: ", student_logits.shape)
            # print("teacher_outputs.logits.shape: ", teacher_outputs.logits.shape)
            # logit_loss = F.kl_div(F.log_softmax(student_logits, dim=1), F.softmax(teacher_outputs.logits, dim=1),
            #                      reduction='batchmean')
            logit_loss = F.kl_div(F.log_softmax(student_logits / args.temperature, dim=1),
                                  F.softmax(teacher_outputs.logits / args.temperature, dim=1), reduction='batchmean') * (
                                     args.temperature ** 2)

            logit_loss_list.append(logit_loss.item())
            # print("logit_loss: ", logit_loss, logit_loss.dtype)

            tea_rep = torch.tensor(np.array([item.cpu().detach().numpy() for item in tea_rep]), dtype=torch.float32)
            tea_rep = tea_rep.to(device=device)

            rep_loss = 0
            tea_rep = tea_rep[args.ignored_layers:]
            stu_rep = stu_rep[args.ignored_layers:]
            # print(len(stu_rep))
            for i in range(len(stu_rep)):
                # print("stu_rep[i]", stu_rep[i])
                # print("tea_rep[i]", tea_rep[i])
                rep_loss += F.mse_loss(stu_rep[i], tea_rep[i])
            rep_loss = rep_loss / batch_size  # batch mean
            rep_loss_list.append(rep_loss.item())
            # print("rep_loss: ", rep_loss)

            total_loss = (args.emb_weight * embeddings_loss) \
                         + (args.ce_weight * ce_loss) \
                         + (args.logit_weight * logit_loss) \
                         + (args.rep_weight * rep_loss)
            # print("total_loss: ", total_loss.item())
            # print("total_loss:{} | ce_loss {} | logit_loss {} | rep_loss {} ".format(total_loss.item(), \
            #     ce_loss.item(), logit_loss.item(), rep_loss))
            total_loss_list.append(total_loss.item())

            optimer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimer)
            scaler.update()
            functional.reset_net(student_model)

            # print(
            #     f"In average, at epoch {epoch}, "
            #     + f"ce_loss: {np.mean(ce_loss_list)}, "
            #     + f"emb_loss: {np.mean(embeddings_loss_list)} "
            #     + f"logit_loss: {np.mean(logit_loss_list)}, "
            #     + f"rep_loss: {np.mean(rep_loss_list)}, "
            #     + f"total_loss: {np.mean(total_loss_list)}"
            # )

        # scheduler.step()
        # scheduler_warmup.step()

        # test_y_true = []
        # test_y_pred = []

        # valid_y_true = []
        # valid_y_pred = []

        student_model.eval()
        with torch.no_grad():
            # Initialize lists to store true and predicted labels for test set
            test_y_true, test_y_pred = [], []

            # Evaluate on test set
            for batch in tqdm(test_data_loader):
                batch_size = len(batch[0])
                b_y = batch[1]
                test_y_true.extend(b_y.to("cpu").tolist())
                inputs = tokenizer(batch[0], padding="max_length", truncation=True, return_tensors='pt',
                                   max_length=args.max_length)
                to_device(inputs, device)
                _, outputs = student_model(inputs['input_ids'])
                outputs = outputs.to("cpu").reshape(-1, args.num_step, args.label_num).transpose(0, 1)
                logits = torch.mean(outputs, dim=0)
                test_y_pred.extend(torch.max(logits, 1)[1].tolist())
                functional.reset_net(student_model)

            # Initialize lists to store true and predicted labels for validation set
            valid_y_true, valid_y_pred = [], []

            # Evaluate on validation set
            for batch in tqdm(valid_data_loader):
                batch_size = len(batch[0])
                b_y = batch[1]
                valid_y_true.extend(b_y.to("cpu").tolist())
                inputs = tokenizer(batch[0], padding="max_length", truncation=True, return_tensors='pt',
                                   max_length=args.max_length)
                to_device(inputs, device)
                _, outputs = student_model(inputs['input_ids'])
                outputs = outputs.to("cpu").reshape(-1, args.num_step, args.label_num).transpose(0, 1)
                logits = torch.mean(outputs, dim=0)
                valid_y_pred.extend(torch.max(logits, 1)[1].tolist())
                functional.reset_net(student_model)

        # Calculate test accuracy
        correct_test = sum([1 for i in range(len(test_y_true)) if test_y_true[i] == test_y_pred[i]])
        test_acc = correct_test / len(test_y_pred)

        # Calculate validation accuracy
        correct_valid = sum([1 for i in range(len(valid_y_true)) if valid_y_true[i] == valid_y_pred[i]])
        valid_acc = correct_valid / len(valid_y_pred)

        # Calculate MCC metrics
        matthews_corrcoef = MatthewsCorrCoef(num_classes=args.label_num, task='multiclass')
        test_mcc = matthews_corrcoef(torch.tensor(test_y_true), torch.tensor(test_y_pred))
        valid_mcc = matthews_corrcoef(torch.tensor(valid_y_true), torch.tensor(valid_y_pred))

        # Print all metrics
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Validation Accuracy: {valid_acc:.4f}")
        print(f"Test MCC: {test_mcc.item():.4f}")
        print(f"Validation MCC: {valid_mcc.item():.4f}")

        # Record metrics
        record = (test_acc, valid_acc, test_mcc.item(), valid_mcc.item())
        metric_list.append(record)
        print(f"Epoch {epoch} Test Acc: {test_acc} Valid Acc: {valid_acc} Test MCC: {test_mcc} Valid MCC: {valid_mcc}")

        # Save model if it's the best so far based on validation MCC
        directory = "saved_models_params/distilled_spikformer"
        if not os.path.exists(directory):
            os.makedirs(directory)
        if valid_mcc >= max(metric_list, key=lambda x: x[3])[3]:
            torch.save(student_model.state_dict(),
                       f"saved_models_params/distilled_spikformer/hyperparam_new_{args.dataset_name}_ic3_epoch{epoch}_test_acc_{test_acc}_valid_acc_{valid_acc}_test_mcc_{test_mcc.item()}_valid_mcc_{valid_mcc.item()}" +
                       f"_num_step_{args.num_step}_lr_{args.fine_tune_lr}_seed_{args.seed}" +
                       f"_batch_size_{args.batch_size}_depths_{args.depths}_max_length_{args.max_length}" +
                       f"_ce_weight_{args.ce_weight}_logit_weight_{args.logit_weight}_rep_weight_{args.rep_weight}" +
                       f"_tau_{args.tau}_common_thr_{args.common_thr}"
                       )
        print("Best validation MCC so far:", max(metric_list, key=lambda x: x[3]))


if __name__ == "__main__":
    _args = parse_args()
    for arg in vars(_args):
        print(arg, getattr(_args, arg))
    set_seed(_args.seed)
    distill(_args)
