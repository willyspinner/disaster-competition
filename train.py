from model import RBTModel
import torch
from tqdm import tqdm
from torch.nn import BCELoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data import get_dataloaders
import csv
from evaluation import evaluate_test, make_prediction_submission


def validate_model(model, dev_loader):
    """
    returns:
    average_val_acc, average_val_loss
    """
    model.eval()
    val_running_loss = 0.0
    running_val_acc = 0.0
    with torch.no_grad():
        # validation loop
        for dev_batch in dev_loader:
            dev_labels = dev_batch['label'].to(device)
            dev_inputs = dev_batch['input']
            val_loss, model_output = model.generate_losses(dev_inputs, dev_labels, loss_criterion)
            predicted_indices = torch.argmax(model_output, dim=1)

            val_correct_pct =  torch.sum(predicted_indices == dev_labels) / len(model_output)
            running_val_acc += val_correct_pct
            val_running_loss += val_loss.item()
    average_val_acc = running_val_acc / len(dev_loader)
    average_val_loss = val_running_loss / len(dev_loader)

            # resetting running values
    return average_val_acc, average_val_loss

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
    # learning rate decay.
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=2, min_lr=1e-7, verbose=True)
    with tqdm(desc="Steps", total=num_epochs * len(train_loader)) as pbar:
        for epoch in range(num_epochs):
            for train_batch in train_loader:
                model.train()
                labels = train_batch['label'].to(device)
                encoded_inputs = train_batch['input']
                loss, model_output = model.generate_losses(encoded_inputs, labels, loss_criterion)
                optimizer.zero_grad()
                loss.backward()

                clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                predicted_indices = torch.argmax(model_output, dim=1)
                correct_pct =  torch.sum(predicted_indices == labels) / len(model_output)
                running_acc += correct_pct

                # update running values
                running_loss += loss.item()
                global_step += 1

                # evaluation step
                if global_step % eval_every == 0:
                    average_val_acc, average_val_loss = validate_model(model, dev_loader)
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
            # end of epoch.

            average_val_acc, average_val_loss = validate_model(model, dev_loader)
            print('Epoch [{}/{}], Step [{}/{}], END OF EPOCH. Valid Loss: {:.4f}, Valid acc: {:.4f}'
                  .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                          average_val_loss, average_val_acc))
            lr_scheduler.step(average_val_loss)



if __name__ == '__main__':
    # Configurations
    device= 'cuda:0' if torch.cuda.is_available() else 'cpu'
    batch_size = 16
    num_epochs = 6
    use_preprocess = True
    lr=1e-5
    eval_every = 100
    print_loss_every = 5
    split=[80, 5, 15] # train, dev, test percentages
    csvpath = "./train.csv"


    model = RBTModel(device=device)
    model_params = model.parameters()
    optimizer = Adam(model_params, lr=lr) # vanilla Adam
    loss_criterion = BCELoss()
    train_loader, dev_loader, test_loader = get_dataloaders(csvpath, device, batch_size, split=split, preprocess=use_preprocess)
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
    test_loss, test_metrics = evaluate_test(model, test_loader, loss_criterion, device)
    model_filename = '{}_loss_{:.5f}_acc_{:.5f}.pt'.format(model.name, test_loss, test_metrics['acc'])
    torch.save(model.state_dict(), model_filename)
    """
    To load the model, simply
    model = RBTModel(device='cuda')
    model.load_state_dict(torch.load(model_filename))
    model.eval()
    """
    print("Model saved at", model_filename)

    make_prediction_submission(model, device)
