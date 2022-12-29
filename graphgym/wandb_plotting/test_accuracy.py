import argparse
import wandb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 20})

# parse configuration
parser = argparse.ArgumentParser(description='GraphGym')

parser.add_argument('--entity', type=str, required=True,
                    help='W&B entity')
parser.add_argument('--project', type=str, required=True,
                    help='W&B project')


parser.add_argument('--main_key', type=str, default='dataset.node_features')
parser.add_argument('--main_labels', type=str, default='none/labels/title/labels,title')
parser.add_argument('--main_texts', type=str, default='no feature/labels/title/labels+title')
parser.add_argument('--sub_key', type=str, default='gnn.layer_type')
parser.add_argument('--sub_labels', type=str, default='gcnconv/gatconv/ginconv')
parser.add_argument('--sub_texts', type=str, default='GCN/GAT/GIN')
parser.add_argument('--train_loss_key', type=str, default='train/loss')
parser.add_argument('--test_accuracy_key', type=str, default='test/classification_multilabel_with_ignore_index.accuracy')
parser.add_argument('--test_accuracy_key_2', type=str, default='test/classification_multilabel')
parser.add_argument('--colors', type=str, default='lightgray/gray/pink/orangered')
parser.add_argument('--subcolors', type=str, default='red/blue/green')
parser.add_argument('--markers', type=str, default='o/*/^')
parser.add_argument('--alphas', type=str, default='0.33/0.66/1')
parser.add_argument('--fig_path', type=str, default='plot.pdf')
parser.add_argument('--bar_width', type=float, default=0.2)
parser.add_argument('--fill_between', action='store_true')
parser.add_argument('--y_top', type=float, default=None)
parser.add_argument('--y_bottom', type=float, default=None)
parser.add_argument('options', default=None, nargs=argparse.REMAINDER)


args = parser.parse_args()


def parse(mapping, key, sep='.'):
    keys = key.split(sep)
    rst = mapping[keys[0]]
    for subkey in keys[1:]:
        rst = rst[subkey]
    return rst


def main():
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2)

    run_groups = {ml: {sl: [] for sl in sub_labels} for ml in main_labels}
    for run in runs:
        try:
            main_label = run.config[main_key]
        except:
            main_label = run.config["dataset"]["node_features"]
        if main_label in run_groups:
            sub_label = run.config[sub_key]
            if sub_label in run_groups[main_label]:
                run_groups[main_label][sub_label].append(run)

    # train loss
    ax = fig.add_subplot(gs[0, 0])
    max_x = -1
    if args.fill_between:
        for i, main_label in enumerate(main_labels):
            xs, ys = [], []
            for j, sub_label in enumerate(sub_labels):
                for run in run_groups[main_label][sub_label]:
                    x, y = [], []
                    for row in run.history(keys=['epoch', args.train_loss_key], pandas=False):
                        x.append(row['epoch'])
                        y.append(row[args.train_loss_key])
                    xs.append(x)
                    ys.append(y)
            for x in xs:
                assert x == xs[0], f'epochs much match in all runs of the same config.'
            x = xs[0]  # epochs of the first run
            max_x = max(max_x, max(x))
            ys = np.stack(ys)
            y, y_err = ys.mean(axis=0), ys.std(axis=0)
            ax.plot(x, y, color=colors[i], marker='.')
            ax.fill_between(x, y+y_err, y-y_err, alpha=0.2, color=colors[i])
    else:
        for i, main_label in enumerate(main_labels):
            for j, sub_label in enumerate(sub_labels):
                xs, ys = [], []
                for run in run_groups[main_label][sub_label]:
                    x, y = [], []
                    for row in run.history(keys=['epoch', args.train_loss_key], pandas=False):
                        x.append(row['epoch'])
                        y.append(row[args.train_loss_key])
                    xs.append(x)
                    ys.append(y)
                for x in xs:
                    assert x == xs[0], f'epochs much match in all runs of the same config.'
                x = xs[0]  # epochs of the first run
                max_x = max(max_x, max(x))
                ys = np.stack(ys)
                y, y_err = ys.mean(axis=0), ys.std(axis=0)
                ax.plot(x, y, color=colors[i], marker=markers[j], alpha=alphas[j])
        for j, sub_text in enumerate(sub_texts):
            ax.plot([], color='k', marker=markers[j], alpha=alphas[j], label=sub_text)
    for i, main_text in enumerate(main_texts):
        ax.plot([], color=colors[i], label=main_text)
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_xlim(left=0, right=max_x)
    ax.set_ylim(bottom=args.y_bottom, top=args.y_top)
    ax.set_ylabel('Training loss')

    # test accuracy
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(sub_labels))
    width = 0.2
    for i, (main_label, main_text) in enumerate(zip(main_labels, main_texts)):
        run_group = run_groups[main_label]
        values = []
        errs = []
        for j, (sub_label, sub_runs) in enumerate(run_group.items()):
            print(main_label, sub_label)
            ys = []
            for run in sub_runs:
                try:
                    ys.append(parse(run.summary, args.test_accuracy_key) * 100)
                except:
                    try:
                        ys.append(run.summary[args.test_accuracy_key]['accuracy'] * 100)
                    except:
                        ys.append(run.summary[args.test_accuracy_key_2]['accuracy'] * 100)
            y, yerr = np.mean(ys), np.std(ys)
            values.append(y)
            errs.append(yerr)
        margin = width * (i + 0.5 - len(sub_labels) * 0.5)
        ax.bar(x + margin, values, yerr=errs, capsize=4, width=width, label=main_text, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(sub_texts)
    ax.set_ylabel('Test accuracy [%]')
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.fig_path, bbox_inches='tight')


if __name__ == '__main__':
    api = wandb.Api()
    filters = [
        {'config.dataset.name': args.dataset},
        {'config.dataset.task': args.task},
        {'config.dataset.task_type': args.task_type},
    ]
    if args.train_sampler is not None:
        filters.append({'config.train.sampler': args.train_sampler})
    if args.layer_type is not None:
        filters.append({'config.gnn.layer_type': args.layer_type})
    if args.batch_size is not None:
        filters.append({'config.train.batch_size': args.batch_size})
    for key, value in zip(args.options[0::2], args.options[1::2]):
        filters.append({key: value})
    filters.append({'state': 'finished'})
    runs = api.runs(f'{args.entity}/{args.project}', filters={'$and': filters})
    main_key = args.main_key
    main_labels = args.main_labels.split('/')
    main_texts = args.main_texts.split('/')
    assert len(main_labels) == len(main_texts)
    sub_key = args.sub_key
    sub_labels = args.sub_labels.split('/')
    sub_texts = args.sub_texts.split('/')
    assert len(sub_labels) == len(sub_texts)
    colors = args.colors.split('/')
    markers = args.markers.split('/')
    alphas = list(map(float, args.alphas.split('/')))
    main()


# plot the test accuracies of each model at the last epoch.
# The test accuracies are stored in different ways for the different models:
    # - all graphgym implementations: 'test_accuracy' is one value
    # - all cwn implementations: it is stored in test_perf, which, however, is tracked during multiple epochs --> need to extract the last epoch
    # - all k-gnn implmentations: stored in test_acc, which is also tracked during multiple epochs.

def extract_test_acc(model: str=None, run: wandb.wandb_run.Run=None):
    if model=="cwn":
        # extract test_perf from run
        return NotImplementedError
    elif model=="k-gnn":
        # extract test_acc from run
        return NotImplementedError


def plot_test_acc():
    '''

    Produces plots of test accuracies of specified models/configurations.

    '''
    return NotImplementedError

