import csv
import torch
from tqdm import tqdm

def compute_f1():
    pass
def compute_precision():
    pass
def compute_recall():
    pass
def compute_auroc():
    pass

def evaluate_test(model, test_loader, loss_criterion, device):
    model.eval()
    total_samples_tested = 0
    total_batches = 0
    correct_samples = 0
    test_loss_sum = 0

    with torch.no_grad():
        # validation loop
        for test_batch in test_loader:
            test_labels = test_batch['label'].to(device)
            # TODO: dont use encoded_text. Just use 'input'
            test_inputs = test_batch['encoded_text']

            test_loss, model_output = model.generate_losses(test_inputs, test_labels, loss_criterion)
            predicted_indices = torch.argmax(model_output, dim=1)

            correct_samples +=  torch.sum(predicted_indices == test_labels)
            total_samples_tested += len(model_output)

            total_batches += 1
            test_loss_sum += test_loss.item()

    test_acc = correct_samples / total_samples_tested
    test_loss = test_loss_sum / total_batches
    print("TEST RESULTS: test_acc: {}, test_loss: {}".format(test_acc, test_loss))
    return test_acc, test_loss 


def make_prediction_submission(model, device):
    model.eval()
    with open('./test.csv') as f:
        # load all text in memory, makes it easier.
        reader = csv.DictReader(f)
        test_rows = [(row['id'], row['text']) for row in reader]
    results = []
    for (id, text) in tqdm(test_rows):
        prediction = model.predict(text).item()
        results.append((id, prediction))

    # generate submission file
    with open('final-results.csv', 'w+') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['id', 'target'])
        writer.writeheader()
        for (id, predicted) in results:
            writer.writerow({ 'id': id, 'target': predicted })

    print("Final submission csv written in final-results.csv")
