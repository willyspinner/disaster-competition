import csv
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score


def get_metrics(labels, preds):
    """
    dict of acc, f1, prec, recl, auroc
    """
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    return {
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "prec": precision_score(labels, preds),
        "recl": recall_score(labels, preds),
        "auroc": roc_auc_score(labels, preds)
    }

def evaluate_test(model, test_loader, loss_criterion, device):
    """
    returns 
    (test_loss, dict{acc, f1, prec, recl, auroc})
    """

    model.eval()
    total_batches = 0
    test_loss_sum = 0
    concat_labels = torch.Tensor([]).to(device)
    concat_preds = torch.Tensor([]).to(device)
    with torch.no_grad():
        # validation loop
        for test_batch in test_loader:
            test_labels = test_batch['label'].to(device)
            test_inputs = test_batch['input']

            test_loss, model_output = model.generate_losses(test_inputs, test_labels, loss_criterion)
            predicted_indices = torch.argmax(model_output, dim=1)

            concat_labels = torch.hstack((concat_labels, test_labels))
            concat_preds = torch.hstack((concat_preds, predicted_indices))
            total_batches += 1
            test_loss_sum += test_loss.item()

    test_loss = test_loss_sum / total_batches
    metrics  = get_metrics(concat_labels, concat_preds)
    print("TEST RESULTS: loss: {:.4f}, acc: {:.4f}, f1: {:.4f}, auroc: {:.4f}, precision: {:.4f}, recall: {:.4f}".format(
        test_loss,
        metrics["acc"],
        metrics["f1"],
        metrics["auroc"],
        metrics["prec"],
        metrics["recl"]))
    return test_loss, metrics

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
