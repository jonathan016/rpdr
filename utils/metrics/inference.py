import torch


def infer(model, data, is_cuda=False):
    model.eval()

    outputs = []
    predictions = []
    ground_truths = []

    for datum, ground_truth in data:
        with torch.no_grad():
            if is_cuda:
                datum, ground_truth = datum.cuda(), ground_truth.cuda()

            output = model(datum)
            _, prediction = torch.max(output.data, 1)

            outputs.append(output.tolist())
            predictions.append(prediction.item())
            ground_truths.append(ground_truth.item())

            del datum, ground_truth, output, prediction
            if is_cuda:
                torch.cuda.empty_cache()

    return outputs, predictions, ground_truths
