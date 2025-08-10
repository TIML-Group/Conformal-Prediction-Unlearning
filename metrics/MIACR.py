import numpy as np
import torch
import torch.nn.functional as F
from sklearn.svm import SVC
from torchvision import transforms


def get_x_y_from_data_dict(data, device):
    x, y = data.values()
    if isinstance(x, list):
        x, y = x[0].to(device), y[0].to(device)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)

def m_entropy(p, labels, dim=-1, keepdim=False):
    log_prob = torch.where(p > 0, p.log(), torch.tensor(1e-30).to(p.device).log())
    reverse_prob = 1 - p
    log_reverse_prob = torch.where(
        p > 0, p.log(), torch.tensor(1e-30).to(p.device).log()
    )
    modified_probs = p.clone()
    modified_probs[:, labels] = reverse_prob[:, labels]
    modified_log_probs = log_reverse_prob.clone()
    modified_log_probs[:, labels] = log_prob[:, labels]
    return -torch.sum(modified_probs * modified_log_probs, dim=dim, keepdim=keepdim)

def collect_prob(dl, model, device):
    if dl is None:
        return torch.zeros([0, 10]), torch.zeros([0])

    prob = []
    targets = []

    model.eval()
    for data, target in dl:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
            prob.append(F.softmax(output, dim=-1).data)
            targets.append(target)

    return torch.cat(prob), torch.cat(targets)


def SVC_fit_predict(shadow_train, shadow_test, target_train, target_test):
    n_shadow_train = shadow_train.shape[0]
    n_shadow_test = shadow_test.shape[0]
    n_target_train = target_train.shape[0]
    n_target_test = target_test.shape[0]

    X_shadow = (
        torch.cat([shadow_train, shadow_test])
        .cpu()
        .numpy()
        .reshape(n_shadow_train + n_shadow_test, -1)
    )
    Y_shadow = np.concatenate([np.ones(n_shadow_train).astype(int), np.zeros(n_shadow_test).astype(int)])

    clf = SVC(C=3, gamma="auto", kernel="rbf", probability=True)
    clf.fit(X_shadow, Y_shadow)

    accs = []

    if n_target_train > 0:
        X_target_train = target_train.cpu().numpy().reshape(n_target_train, -1)
        acc_train = clf.predict(X_target_train).mean()
        accs.append(acc_train)
        # proba_train = clf.predict_proba(X_target_train)

    if n_target_test > 0:
        X_target_test = target_test.cpu().numpy().reshape(n_target_test, -1)
        acc_test = 1 - clf.predict(X_target_test).mean()
        accs.append(acc_test)
        # proba_test = clf.predict_proba(X_target_test)

    return np.mean(accs)

def find_quantile(cal_scores, n, alpha):
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    qhat = np.quantile(cal_scores, q_level, method='higher')
    return qhat

def SVC_fit_predict_cp(shadow_train, shadow_test, target_train, target_test, cal_ds):

    n_shadow_train = shadow_train.shape[0]
    n_shadow_test = shadow_test.shape[0]
    n_target_train = target_train.shape[0]
    n_target_test = target_test.shape[0]
    n_cal = cal_ds.shape[0]

    X_shadow = (
        torch.cat([shadow_train, shadow_test])
        .cpu()
        .numpy()
        .reshape(n_shadow_train + n_shadow_test, -1)
    )
    Y_shadow = np.concatenate([np.ones(n_shadow_train).astype(int), np.zeros(n_shadow_test).astype(int)])

    clf = SVC(C=3, gamma="auto", kernel="rbf", probability=True)
    clf.fit(X_shadow, Y_shadow)

    '''calibration process'''
    X_cal_cp = cal_ds.cpu().numpy().reshape(n_cal, -1)
    cal_proba = clf.predict_proba(X_cal_cp)
    cal_scores = 1 - cal_proba[np.arange(n_cal), np.ones(n_cal).astype(int)]
    q_hat = find_quantile(cal_scores, n_cal, 0.05)

    '''test set'''
    acc_train, coverage_train, set_size_train = CP(target_train, q_hat, 1, clf)
    if n_target_test > 0:
        acc_test, coverage_test, set_size_test = CP(target_train, q_hat, 0, clf)
        acc_test = 1-acc_test
        return acc_train, coverage_train, set_size_train, q_hat, acc_test, coverage_test, set_size_test
    return acc_train, coverage_train, set_size_train, q_hat


def CP(target_data, q_hat, flag, clf):
    n_target_data = target_data.shape[0]
    X_target_data = target_data.cpu().numpy().reshape(n_target_data, -1)
    acc = clf.predict(X_target_data).mean()
    '''conformal prediction'''
    proba_train = clf.predict_proba(X_target_data)
    prediction_train_sets = proba_train >= (1-q_hat)
    set_size = prediction_train_sets.sum(axis=1).mean()
    coverage_possibility = prediction_train_sets[
        np.arange(n_target_data),
        np.ones(n_target_data).astype(int) if flag==1 else np.zeros(n_target_data).astype(int)
    ].mean()
    coverage_possibility = 1-coverage_possibility
    return acc, coverage_possibility, set_size

def SVC_MIA(shadow_train, shadow_test, target_train, target_test, cal_dl, model, device):
    shadow_train_prob, shadow_train_labels = collect_prob(shadow_train, model, device)
    shadow_test_prob, shadow_test_labels = collect_prob(shadow_test, model, device)
    target_train_prob, target_train_labels = collect_prob(target_train, model, device)
    target_test_prob, target_test_labels = collect_prob(target_test, model, device)
    cal_prob, cal_labels = collect_prob(cal_dl, model, device)

    shadow_train_conf = torch.gather(shadow_train_prob, 1, shadow_train_labels[:, None])
    shadow_test_conf = torch.gather(shadow_test_prob, 1, shadow_test_labels[:, None])
    target_train_conf = torch.gather(target_train_prob, 1, target_train_labels[:, None])
    target_test_conf = torch.gather(target_test_prob, 1, target_test_labels[:, None])
    cal_ds_conf = torch.gather(cal_prob, 1, cal_labels[:, None])

    acc_conf = SVC_fit_predict_cp(
        shadow_train_conf, shadow_test_conf, target_train_conf, target_test_conf, cal_ds_conf
    )
    return acc_conf


def dataset_convert_to_test(dataset):
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    dataset.transform = test_transform
    dataset.train = False