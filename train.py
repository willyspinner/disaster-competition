from model import Model
import torch
from torch.nn import BCELoss
from torch.optim import Adam
from data import get_data_iterators
import torch.nn.functional as F

# adapted from existing code

def train_model(model, optimizer, loss_criterion, num_epochs, eval_every):
 # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for (text, labels), _ in train_loader:
            labels = F.one_hot(labels, num_classes=2).type(torch.float)
            labels = labels.to(device)
            text = text.type(torch.LongTensor)
            text = text.to(device)
            model_output = model.forward(text)
            # model_output is (batch_size, 2)
            loss = loss_criterion(model_output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                """
                model.eval()
                with torch.no_grad():

                    # validation loop
                    for (labels, text), _ in valid_loader:
                        labels = labels.type(torch.LongTensor)
                        labels = labels.to(device)
                        text = text.type(torch.LongTensor)
                        text = text.to(device)
                        output = model(text, labels)
                        loss, _ = output

                        valid_running_loss += loss.item()
                """

                # evaluation
                average_train_loss = running_loss / eval_every
                #average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                #valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                #valid_running_loss = 0.0
                model.train()

                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss))
                # print progress
                """
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss))

                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    print("BEST VALID LOSS", best_valid_loss)
                    #save_checkpoint(file_path + '/' + 'model.pt', model, best_valid_loss)
                    #save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
                """


if __name__ == '__main__':
    # Configurations
    device='cpu'
    batch_size = 16
    num_epochs = 5
    eval_every = 1

    model = Model()
    optimizer = Adam(model.parameters()) # vanilla Adam
    loss_criterion = BCELoss()
    train_loader, test_loader = get_data_iterators(device, batch_size)
    train_model(
        model,
        optimizer=optimizer,
        loss_criterion=loss_criterion,
        num_epochs=num_epochs,
        eval_every=eval_every
    )
