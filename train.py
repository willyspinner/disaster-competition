from model import Model
import torch
from tqdm import tqdm
from torch.nn import BCELoss
from torch.optim import Adam
from data import get_dataloaders
import torch.nn.functional as F

# adapted from existing code


def generate_losses(model, data_batch, device, loss_criterion):
    labels = data_batch['label']
    encoded_texts = data_batch['encoded_text']

    input_ids = encoded_texts['input_ids'].to(device)
    attn_mask = encoded_texts['attention_mask'].to(device)

    labels = labels.to(device)
    labels_one_hot = F.one_hot(labels, num_classes=2).type(torch.float).to(device)

    model_output = model.forward(input_ids, attention_mask=attn_mask).to(device)
    loss = loss_criterion(model_output, labels_one_hot)

    return loss, model_output

def train_model(model, optimizer, loss_criterion, num_epochs, eval_every, print_loss_every, train_loader, dev_loader, device):
 # initialize running values
    running_acc = 0.0
    running_loss = 0.0
    val_running_loss = 0.0
    best_val_loss = float('inf')
    global_step = 0
    train_loss_list = []
    val_loss_list = []

    # training loop
    model.train()
    with tqdm(desc="Steps", total=num_epochs * len(train_loader)) as pbar:
        for epoch in range(num_epochs):
            for train_batch in train_loader:
                labels = train_batch['label'].to(device)
                loss, model_output = generate_losses(model, train_batch, device, loss_criterion)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                predicted_indices = torch.argmax(model_output, dim=1)
                correct_pct =  torch.sum(predicted_indices == labels) / len(model_output)
                running_acc += correct_pct

                # update running values
                running_loss += loss.item()
                global_step += 1

                # evaluation step
                if global_step % eval_every == 0:
                    model.eval()
                    running_val_acc = 0
                    with torch.no_grad():
                        # validation loop
                        for dev_batch in dev_loader:
                            dev_labels = dev_batch['label'].to(device)
                            val_loss, model_output = generate_losses(model, dev_batch, device, loss_criterion)
                            predicted_indices = torch.argmax(model_output, dim=1)

                            val_correct_pct =  torch.sum(predicted_indices == dev_labels) / len(model_output)
                            running_val_acc += val_correct_pct

                            val_running_loss += val_loss.item()
                    # evaluation
                    average_val_acc = running_val_acc / len(dev_loader)
                    average_val_loss = val_running_loss / len(dev_loader)
                    val_loss_list.append(average_val_loss)

                    # resetting running values
                    running_val_acc = 0.0
                    val_running_loss = 0.0
                    model.train()

                    print('Epoch [{}/{}], Step [{}/{}], ON VALIDATION SET: Valid Loss: {:.4f}, Valid acc: {:.4f}'
                          .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                                  average_val_loss, average_val_acc))
                    # checkpoint
                    if best_val_loss > average_val_loss:
                        best_val_loss = average_val_loss
                        print("BEST VALID LOSS", best_val_loss)
                        #save_checkpoint(file_path + '/' + 'model.pt', model, best_val_loss)
                        #save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, val_loss_list, global_steps_list)

                if global_step % print_loss_every == 0:
                    average_train_loss = running_loss / print_loss_every
                    train_loss_list.append(average_train_loss)
                    average_train_acc = running_acc / print_loss_every
                    print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Train acc: {:.4f}'
                          .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                                  average_train_loss, average_train_acc))
                    running_loss = 0.0
                    running_acc = 0.0

                pbar.update(1)


if __name__ == '__main__':
    # Configurations
    device= 'cuda:0' if torch.cuda.is_available() else 'cpu'
    batch_size = 16
    num_epochs = 10
    lr=0.00001
    eval_every = 100
    print_loss_every = 5
    split=[80, 5, 15] # train, dev, test

    # TODO: need to decide if we run the dev set every epoch? Or how?

    model = Model(device=device)
    model_params = model.parameters()
    optimizer = Adam(model_params, lr=lr) # vanilla Adam
    loss_criterion = BCELoss()
    train_loader, dev_loader, test_loader = get_dataloaders(device, batch_size)
    train_model(
        model,
        optimizer=optimizer,
        loss_criterion=loss_criterion,
        num_epochs=num_epochs,
        eval_every=eval_every,
        print_loss_every=print_loss_every,
        train_loader=train_loader,
        dev_loader=dev_loader,
        device=device
    )
