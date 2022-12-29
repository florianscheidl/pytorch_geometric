import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# parse configuration
def test_acc_plots(df,dataset, graph: bool = True):
    # g = sns.FacetGrid(data=df, col="type")
    # g.map(sns.boxplot, data=df, x="method", y="test_acc")


    methods = list(df["method"])
    k_gnn = "3-gnn" if "3-gnn" in methods else "2-gnn"
    sin = "sin-clique-2" if "sin-clique-2" in methods else "sin-clique-3"
    cin = "cin-rings-5" if "cin-rings-5" in methods else "cin-rings-8"
    het_rings = "-rings-8" if "hgt-rings-8" in methods else "-rings-5"
    het_dim = "-clique-2" if "hgt-clique-2" in methods else "-clique-3"

    het_methods = ["han","hgt","het"]
    if graph:
        order = ["gcnconv","ginconv","sageconv","gatconv",k_gnn,sin,cin]+[het_method+het_dim for het_method in het_methods]+[het_method+het_rings for het_method in het_methods]
    elif dataset != 'pubmed':
        order = ["gcnconv", "ginconv", "sageconv", "gatconv"] + [het_method + het_dim for het_method in het_methods] + [
            het_method + het_rings for het_method in het_methods]
    else:
        order = ["gcnconv", "ginconv", "sageconv", "gatconv"] + [het_method + het_dim for het_method in het_methods]

    rc_dict = {'font.size': 14,
               'axes.labelsize': "medium",
               'boxplot.meanline' : True,
               'axes.titlesize': 18,
               'axes.titleweight': "medium"}
    sns.set_theme(style="ticks",palette="Dark2",rc=rc_dict)
    # sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    plt.figure()
    sns.boxplot(data=df, y="test_acc", x="method",hue="type", order=order, dodge=False, width=0.6)
    plt.xticks(rotation="60")

    if dataset=="mutag":
        plt.ylim([0.4,1])
    elif dataset=="imdb-b":
        plt.ylim([0.55, 0.825])
    elif dataset=="imdb-m":
        plt.ylim([0.37, 0.6])
    elif dataset=="nci109":
        plt.ylim([0.57,0.87])
    elif dataset=="proteins":
        plt.ylim([0.52,0.81])

    plt.yticks()
    plt.legend(loc="lower center", ncol=2) #
    plt.ylabel("Test accuracy")
    plt.xlabel(None)
    plt.title(dataset.upper())

    plt.tight_layout()
    # plt.show()
    plt.savefig("/home/flo/MastersThesis/figures/"+dataset+"_test_acc_plot", dpi=300)

    # legend and order to be changed


def curve_plots(df, dataset, curves : str = None, save : bool = False, graph : bool = True):

    methods = list(df["method"])
    k_gnn = "3-gnn" if "3-gnn" in methods else "2-gnn"
    sin = "sin-clique-2" if "sin-clique-2" in methods else "sin-clique-3"
    cin = "cin-rings-5" if "cin-rings-5" in methods else "cin-rings-8"
    het_rings = "-rings-8" if "hgt-rings-8" in methods else "-rings-5"
    het_dim = "-clique-2" if "hgt-clique-2" in methods else "-clique-3"

    het_methods = ["han", "hgt", "het"]
    if graph:
        order = ["gcnconv", "ginconv", "sageconv", "gatconv", k_gnn, sin, cin] + [het_method + het_dim for het_method in
                                                                              het_methods] + [het_method + het_rings for
                                                                                              het_method in het_methods]
    elif dataset!='pubmed':
        order = ["gcnconv","ginconv","sageconv","gatconv"]+[het_method+het_dim for het_method in het_methods]+[het_method+het_rings for het_method in het_methods]
    else:
        order = ["gcnconv","ginconv","sageconv","gatconv"]+[het_method+het_dim for het_method in het_methods]



    rc_dict = {'font.size': 18,
               'axes.labelsize': "medium",
               'xtick.labelsize': 16,
               'xtick.minor.visible': True,
               'boxplot.meanline': True,
               'axes.titlesize': 18,
               'axes.titleweight': "medium"}
    sns.set_theme(style="ticks", palette="Dark2", rc=rc_dict)
    # sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    plt.figure()

    # sns.boxplot(data=df, y="test_acc", x="method", hue="type", order=order, dodge=False, width=0.6)
    # sns.lineplot(data=df, x="epoch", y="val_loss", )

    if graph:
        col_order = ['simple graph', 'node-tuple', 'simplicial complex', 'cell complex']
        col_wrap = 2
    elif dataset != 'pubmed':
        col_order = ['simple graph', 'simplicial complex', 'cell complex']
        col_wrap = 3
    else:
        col_order = ['simple graph', 'simplicial complex']
        col_wrap = 2

    if curves == 'train':
        g = sns.relplot(kind="line",
                    data=df,
                    x="epoch",
                    y="train_loss",
                    col="type",
                    col_wrap=col_wrap,
                    hue="method",
                    palette="deep",
                    #legend='brief',
                    facet_kws={'sharex': False, 'sharey': True,"legend_out":False},
                    markers=False,dashes=False,
                    col_order=col_order,
                    hue_order=order)
        g.set_axis_labels("Epochs", "Training loss")
    elif curves == 'val':
        g = sns.relplot(kind="line",
                    data=df,
                    x="epoch",
                    y="val_loss",
                    col="type",
                    col_wrap=col_wrap,
                    hue="method",
                    palette="deep",
                    #legend='brief',
                    facet_kws={'sharex': False, 'sharey': True,"legend_out":False},
                    markers=False,dashes=False,
                    col_order=col_order,
                    hue_order=order)
        g.set_axis_labels("Epochs", "Validation loss")

    g.fig.suptitle(dataset.upper())
    g.set_titles("{col_name}".capitalize())

    # labels_list = [[g.legend.texts[i]._text for i in manual_indices[i]] for i in range(len(manual_indices))]
    if graph:
        indxs = [[0,1,2,3],[5,9,7,8], [6,10,11,12]]
    elif dataset!='pubmed':
        indxs = [[0,1,2,3],[4,5,6],[7,8,9]]
    else:
        indxs = [[0, 1, 2, 3], [4, 5, 6]]
    handles, labels = list(g.axes_dict.values())[0].get_legend_handles_labels()
    for i, ax in enumerate(g.axes_dict.values()):
        handles_i = [handles[j] for j in indxs[i]]
        labels_i = [labels[j] for j in indxs[i]]
        ax.legend(handles=handles_i, labels=labels_i)
        ax.yaxis.grid(True)
        ax.xaxis.grid(False)

    #g = sns.FacetGrid(data=df,col="type", hue="method")
    #g.map(sns.lineplot,"epoch","train_loss","method")
    if curves == 'train':
        if dataset == "mutag":
            plt.ylim([0, 1.5])
        elif dataset == "imdb-b":
            plt.ylim([0, 2])
        elif dataset == "imdb-m":
            plt.ylim([0, 8])
        elif dataset == "nci1":
            plt.ylim([0, 0.8])
        elif dataset == "nci109":
            plt.ylim([0, 0.8])
        elif dataset == "proteins":
            plt.ylim([0, 0.8])

    elif curves == 'val':
        if dataset == "mutag":
            plt.ylim([0 , 2])
        elif dataset == "imdb-b":
            plt.ylim([0, 2])
        elif dataset == "imdb-m":
            plt.ylim([0, 6])
        elif dataset == "nci1":
            plt.ylim([0, 0.8])
        elif dataset == "nci109":
            plt.ylim([0, 1])
        elif dataset == "proteins":
            plt.ylim([0, 1])

    plt.tight_layout()
    # plt.show(dpi=300)
    if save:
        if curves == 'train':
            plt.savefig("/home/flo/MastersThesis/figures/" + dataset + "_train_loss_curves", dpi=300)
        elif curves == 'val':
            plt.savefig("/home/flo/MastersThesis/figures/" + dataset + "_val_loss_curves", dpi=300)
    else:
        plt.show()

    plt.close()

# datasets = ["proteins"]
datasets = ["mutag","imdb-b","imdb-m","nci1","nci109","proteins"]
node_datasets = ["citeseer","cora","pubmed"]
# node_datasets = ["cora"]
# datasets = ["mutag"]

accuracy_table_list_graph_mean = []
accuracy_table_list_graph_std = []

for dataset in tqdm(datasets):
    test_acc_df = pd.read_csv(dataset + "-final_test_acc.csv")
    # test_acc_plots(test_acc_df,dataset, graph=False)
    # curves_df = pd.read_csv(dataset+"-final_loss_curves.csv")
    # curves_df=curves_df[curves_df["seed"].isin([10,20])]

    # curve_plots(curves_df, curves='train', dataset=dataset, save=True, graph=False)
    # curve_plots(curves_df, curves='val', dataset=dataset, save=True, graph=False)

    # create accuracy tables
    pd.options.display.float_format = '{:,.2%}'.format
    new_df_mean = test_acc_df.groupby(['method']).mean().reset_index()[['method', 'test_acc']]
    new_df_std = test_acc_df.groupby(['method']).std().reset_index()[['method', 'test_acc']]
    new_df_mean.set_index("method", inplace=True)
    new_df_std.set_index("method", inplace=True)
    new_df_mean.rename(columns={"test_acc": dataset}, inplace=True)
    new_df_std.rename(columns={"test_acc": dataset}, inplace=True)
    accuracy_table_list_graph_mean.append(new_df_mean)
    accuracy_table_list_graph_std.append(new_df_std)

accuracy_table_graph_mean = pd.concat(accuracy_table_list_graph_mean, axis=1)
# accuracy_table_node_mean = accuracy_table_node_mean.astype(str)
accuracy_table_graph_std = pd.concat(accuracy_table_list_graph_std, axis=1)
# accuracy_table_node_std = accuracy_table_node_std.astype(str)

# accuracy_table_node_mean_str = accuracy_table_node_mean.to_string()
accuracy_table_graph_std = accuracy_table_graph_std.rename(index={'gcnconv': 'GCN', 'sageconv': 'GraphSage', 'ginconv': 'GIN',
                                        'gatconv': 'GAT', 'sin-clique-2': 'SIN', 'cin-rings-8': 'CIN',
                                        '3-gnn': 'k-gnn', '2-gnn': 'k-gnn',
                                        'han-clique-3': 'han-clique','han-clique-2': 'han-clique',
                                        'hgt-clique-3': 'hgt-clique','hgt-clique-2': 'hgt-clique',
                                        'het-clique-3': 'het-clique','het-clique-2': 'het-clique',
                                                                    'han-rings-8': 'han-rings',
                                                                    'hgt-rings-8': 'hgt-rings',
                                                                    'het-rings-8': 'het-rings',
                                        })
accuracy_table_graph_mean = accuracy_table_graph_mean.rename(index={'gcnconv': 'GCN', 'sageconv': 'GraphSage', 'ginconv': 'GIN',
                                        'gatconv': 'GAT', 'sin-clique-2': 'SIN', 'cin-rings-8': 'CIN',
                                        '3-gnn': 'k-gnn', '2-gnn': 'k-gnn',
                                        'han-clique-3': 'han-clique','han-clique-2': 'han-clique',
                                        'hgt-clique-3': 'hgt-clique','hgt-clique-2': 'hgt-clique',
                                        'het-clique-3': 'het-clique','het-clique-2': 'het-clique',
                                                                    'han-rings-8': 'han-rings',
                                                                    'hgt-rings-8': 'hgt-rings',
                                                                    'het-rings-8': 'het-rings',
                                        })

accuracy_table_graph_std = accuracy_table_graph_std.sum(level=0)
accuracy_table_graph_mean = accuracy_table_graph_mean.sum(level=0)

mean_std_separated_df_list = []

for col in accuracy_table_graph_mean.columns:
    accuracy_table_graph_mean[col] = accuracy_table_graph_mean[col].map('{:,.2%}'.format)
    accuracy_table_graph_std[col] = accuracy_table_graph_std[col].map('{:,.2%}'.format)
    accuracy_table_graph_std[col] = u"\u00B1"+accuracy_table_graph_std[col].astype(str)
    accuracy_table_graph_mean[col] = accuracy_table_graph_mean[col].astype(str)

    accuracy_table_graph_mean[col] = accuracy_table_graph_mean[col].str.replace("%", "")
    accuracy_table_graph_std[col] = accuracy_table_graph_std[col].str.replace("%", "")


    columns = [(col, 'mean'), (col, 'std')]
    mean_std_separated_df = pd.concat([accuracy_table_graph_mean[col],accuracy_table_graph_std[col]],axis=1)
    mean_std_separated_df.columns = pd.MultiIndex.from_tuples(columns)

    mean_std_separated_df_list.append(mean_std_separated_df)


    # accuracy_table_graph_mean[col] = accuracy_table_graph_mean[col].str.cat(accuracy_table_graph_std[col], sep=u"\u00B1").replace('nan'u"\u00B1"'nan', '')
# accuracy_table_node = accuracy_table_node_mean.cat(accuracy_table_node_std, sep="\plusminus")
full_mean_std_separated_df = pd.concat(mean_std_separated_df_list,axis=1)

# full_mean_std_separated_df = full_mean_std_separated_df.sum(level=0)
full_mean_std_separated_df = full_mean_std_separated_df.reindex(["GCN", "GraphSage", "GIN",'GAT',
                                                               'k-gnn','SIN','CIN',
                                                               'han-clique','hgt-clique','het-clique',
                                                               'han-rings','hgt-rings','het-rings'
                                  ])
full_mean_std_separated_df.to_csv("graph_accuracy.csv")

# accuracy_table_graph_mean = accuracy_table_graph_mean.rename(index={'gcnconv': 'GCN', 'sageconv': 'GraphSage', 'ginconv': 'GIN',
#                                         'gatconv': 'GAT', 'sin-clique-2': 'SIN', 'cin-rings-8': 'CIN',
#                                         '3-gnn': 'k-gnn', '2-gnn': 'k-gnn',
#                                         'han-clique-3': 'han-clique','han-clique-2': 'han-clique',
#                                         'hgt-clique-3': 'hgt-clique','hgt-clique-2': 'hgt-clique',
#                                         'het-clique-3': 'het-clique','het-clique-2': 'het-clique',
#                                                                     'han-rings-8': 'han-rings',
#                                                                     'hgt-rings-8': 'hgt-rings',
#                                                                     'het-rings-8': 'het-rings',
#                                         })
# accuracy_table_graph_mean = accuracy_table_graph_mean.sum(level=0)
# accuracy_table_graph_mean = accuracy_table_graph_mean.reindex(["GCN", "GraphSage", "GIN",'GAT',
#                                                                'k-gnn','SIN','CIN',
#                                                                'han-clique','hgt-clique','het-clique',
#                                                                'han-rings','hgt-rings','het-rings'
#                                   ])
# accuracy_table_graph_mean.to_csv("graph_accuracy.csv")
#accuracy_table_graph.to_csv("graph_accuracy.csv", format)



#plt.savefig("mygraph.png")

accuracy_table_list_node_mean = []
accuracy_table_list_node_std = []

for dataset in tqdm(node_datasets):
    test_acc_df = pd.read_csv(dataset+"-final_test_acc.csv")
    # test_acc_plots(test_acc_df,dataset, graph=False)
    # curves_df = pd.read_csv(dataset+"-final_loss_curves.csv")
    # curves_df=curves_df[curves_df["seed"].isin([10,20])]

    # curve_plots(curves_df, curves='train', dataset=dataset, save=True, graph=False)
    # curve_plots(curves_df, curves='val', dataset=dataset, save=True, graph=False)

    # create accuracy tables
    pd.options.display.float_format = '{:,.2%}'.format
    new_df_mean = test_acc_df.groupby(['method']).mean().reset_index()[['method','test_acc']]
    new_df_std = test_acc_df.groupby(['method']).std().reset_index()[['method','test_acc']]
    new_df_mean.set_index("method", inplace=True)
    new_df_std.set_index("method", inplace=True)
    new_df_mean.rename(columns={"test_acc": dataset}, inplace=True)
    new_df_std.rename(columns={"test_acc": dataset}, inplace=True)
    accuracy_table_list_node_mean.append(new_df_mean)
    accuracy_table_list_node_std.append(new_df_std)

accuracy_table_node_mean = pd.concat(accuracy_table_list_node_mean,axis=1)
# accuracy_table_node_mean = accuracy_table_node_mean.astype(str)
accuracy_table_node_std = pd.concat(accuracy_table_list_node_std,axis=1)
# accuracy_table_node_std = accuracy_table_node_std.astype(str)

#accuracy_table_node_mean_str = accuracy_table_node_mean.to_string()
mean_std_separated_df_list = []

for col in accuracy_table_node_mean.columns:
    accuracy_table_node_mean[col] = accuracy_table_node_mean[col].map('{:,.2%}'.format)
    accuracy_table_node_std[col] = accuracy_table_node_std[col].map('{:,.2%}'.format)
    accuracy_table_node_std[col] = u"\u00B1"+accuracy_table_node_std[col].astype(str)
    accuracy_table_node_mean[col] = accuracy_table_node_mean[col].astype(str)


    accuracy_table_node_mean[col] = accuracy_table_node_mean[col].str.replace("%","")
    accuracy_table_node_std[col] = accuracy_table_node_std[col].str.replace("%","")

    columns = [(col, 'mean'), (col, 'std')]
    mean_std_separated_df = pd.concat([accuracy_table_node_mean[col],accuracy_table_node_std[col]],axis=1)
    mean_std_separated_df.columns = pd.MultiIndex.from_tuples(columns)

    mean_std_separated_df_list.append(mean_std_separated_df)

    accuracy_table_node_mean[col] = accuracy_table_node_mean[col].str.cat(accuracy_table_node_std[col], sep=u"\u00B1").replace('nan'u"\u00B1"'nan', '-')
# accuracy_table_node = accuracy_table_node_mean.cat(accuracy_table_node_std, sep="\plusminus")

full_mean_std_separated_df = pd.concat(mean_std_separated_df_list,axis=1)

full_mean_std_separated_df = full_mean_std_separated_df.rename(index={'gcnconv': 'GCN', 'sageconv': 'GraphSage', 'ginconv': 'GIN',
                                        'gatconv': 'GAT',
                                        'han-clique-3': 'han-clique',
                                        'hgt-clique-3': 'hgt-clique',
                                        'het-clique-3': 'het-clique',
                                        'han-rings-5': 'han-rings',
                                        'hgt-rings-5': 'hgt-rings',
                                        'het-rings-5': 'het-rings',
                                        })
full_mean_std_separated_df = full_mean_std_separated_df.reindex(["GCN", "GraphSage", "GIN",'GAT',
                                  'han-clique','hgt-clique','het-clique',
                                  'han-rings','hgt-rings','het-rings'
                                  ])
full_mean_std_separated_df.to_csv("node_accuracy.csv")

# accuracy_table_node_mean = accuracy_table_node_mean.rename(index={'gcnconv': 'GCN', 'sageconv': 'GraphSage', 'ginconv': 'GIN',
#                                         'gatconv': 'GAT',
#                                         'han-clique-3': 'han-clique',
#                                         'hgt-clique-3': 'hgt-clique',
#                                         'het-clique-3': 'het-clique',
#                                         'han-rings-5': 'han-rings',
#                                         'hgt-rings-5': 'hgt-rings',
#                                         'het-rings-5': 'het-rings',
#                                         })
# accuracy_table_node_mean = accuracy_table_node_mean.reindex(["GCN", "GraphSage", "GIN",'GAT',
#                                   'han-clique','hgt-clique','het-clique',
#                                   'han-rings','hgt-rings','het-rings'
#                                   ])
# accuracy_table_node_mean.to_csv("node_accuracy.csv")
