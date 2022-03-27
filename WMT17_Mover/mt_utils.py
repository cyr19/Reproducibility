import glob

from io import StringIO
import pandas as pd
import os
from scipy.stats import pearsonr, spearmanr, kendalltau
import numpy as np


def find_corpus(name):
    WMT2015 = dict({
        "newstest2015-csen-ref.en": "cs-en",
        "newstest2015-deen-ref.en": "de-en",
        "newstest2015-fien-ref.en": "fi-en",
        "newstest2015-ruen-ref.en": "ru-en"
    })

    WMT2016 = dict({
        "newstest2016-csen-ref.en": "cs-en",
        "newstest2016-deen-ref.en": "de-en",
        "newstest2016-fien-ref.en": "fi-en",
        "newstest2016-ruen-ref.en": "ru-en",
        "newstest2016-roen-ref.en": "ro-en",
        "newstest2016-tren-ref.en": "tr-en",

    })


    WMT2017 = dict({
        "newstest2017-csen-ref.en": "cs-en",
        "newstest2017-deen-ref.en": "de-en",
        "newstest2017-fien-ref.en": "fi-en",
        "newstest2017-lven-ref.en": "lv-en",
        "newstest2017-ruen-ref.en": "ru-en",
        "newstest2017-tren-ref.en": "tr-en",
        "newstest2017-zhen-ref.en": "zh-en"
    })

    WMT2018 = dict({
        "newstest2018-csen-ref.en": "cs-en",
        "newstest2018-deen-ref.en": "de-en",
        "newstest2018-eten-ref.en": "et-en",
        "newstest2018-fien-ref.en": "fi-en",
        "newstest2018-ruen-ref.en": "ru-en",
        "newstest2018-tren-ref.en": "tr-en",
        "newstest2018-zhen-ref.en": "zh-en",
    })
    '''
    WMT2018 = dict({

        "newstest2018-ruen-ref.en": "ru-en"

    })
    '''
    WMT2019 = dict({
        "newstest2019-deen-ref.en": "de-en",
        "newstest2019-fien-ref.en": "fi-en",
        "newstest2019-guen-ref.en": "gu-en",
        "newstest2019-kken-ref.en": "kk-en",

        "newstest2019-lten-ref.en": "lt-en",
        "newstest2019-ruen-ref.en": "ru-en",
        "newstest2019-zhen-ref.en": "zh-en"

    })

    '''
    WMT2019 = dict({

        "newstest2019-kken-ref.en": "kk-en",
        "newstest2019-zhen-ref.en": "zh-en"

    })
    '''
    WMT2020 = dict({
        # "newstest2020-defr-ref.fr.txt": "de-fr",
        "newstest2020-encs-ref.cs.txt": "cs-en",
        "newstest2020-ende-ref.de.txt": "de-en",
        # "newstest2020-eniu-ref.iu.txt": "en-iu",
        "newstest2020-enja-ref.ja.txt": "ja-en",
        # "newstest2020-enkm-ref.km.txt": "en-km",
        "newstest2020-enpl-ref.pl.txt": "pl-en",
        # "newstest2020-enps-ref.ps.txt": "en-ps",
        "newstest2020-enru-ref.ru.txt": "ru-en",
        "newstest2020-enta-ref.ta.txt": "ta-en",
        "newstest2020-enzh-ref.zh.txt": "zh-en"
        # "newstest2020-frde-ref.de.txt": "fr-de"

    })
    if name == 'WMT15':
        dataset = WMT2015
    if name == 'WMT16':
        dataset = WMT2016
    if name == 'WMT17':
        dataset = WMT2017
    if name == 'WMT18':
        dataset = WMT2018
    if name == 'WMT19':
        dataset = WMT2019
    if name == 'WMT20':
        dataset = WMT2020
    return dataset


def find_corpus_foreign(name):
    WMT2017 = {
        "newstest2017-encs-ref.cs": "en-cs",
        "newstest2017-ende-ref.de": "en-de",
        "newstest2017-enfi-ref.fi": "en-fi",
        "newstest2017-enlv-ref.lv": "en-lv",

        "newstest2017-enru-ref.ru": "en-ru",
        "newstest2017-entr-ref.tr": "en-tr",
        "newstest2017-enzh-ref.zh": "en-zh"
    }

    WMT2018 = {
        "newstest2018-encs-ref.cs": "en-cs",
        "newstest2018-ende-ref.de": "en-de",
        "newstest2018-enfi-ref.fi": "en-fi",
        "newstest2018-enru-ref.ru": "en-ru",
        "newstest2018-entr-ref.tr": "en-tr",
        "newstest2018-enzh-ref.zh": "en-zh",
        "newstest2018-enet-ref.et": "en-et"
    }

    WMT2019 = {

        "newstest2019-enfi-ref.fi": "en-fi",
        "newstest2019-engu-ref.gu": "en-gu",
        "newstest2019-enkk-ref.kk": "en-kk",
        "newstest2019-enlt-ref.lt": "en-lt",
        "newstest2019-enru-ref.ru": "en-ru",
        "newstest2019-enzh-ref.zh": "en-zh",
        "newstest2019-ende-ref.de": "en-de"


    }

    WMT2020 = dict({
        # "newstest2020-defr-ref.fr.txt": "de-fr",
        "newstest2020-encs-ref.cs.txt": "en-cs",
        "newstest2020-ende-ref.de.txt": "en-de",
        # "newstest2020-eniu-ref.iu.txt": "en-iu",
        "newstest2020-enja-ref.ja.txt": "en-ja",
        # "newstest2020-enkm-ref.km.txt": "en-km",
        "newstest2020-enpl-ref.pl.txt": "en-pl",
        # "newstest2020-enps-ref.ps.txt": "en-ps",
        "newstest2020-enru-ref.ru.txt": "en-ru",
        "newstest2020-enta-ref.ta.txt": "en-ta",
        "newstest2020-enzh-ref.zh.txt": "en-zh"
        # "newstest2020-frde-ref.de.txt": "fr-de"

    })

    if name == 'WMT17':
        dataset = WMT2017
    if name == 'WMT18':
        dataset = WMT2018
    if name == 'WMT19':
        dataset = WMT2019
    if name == 'WMT20':
        dataset = WMT2020
    return dataset


def load_data(path):
    lines = []
    with open(path, 'r') as f:
        for line in f.readlines():
            l = line.strip()
            lines.append(l)
    return lines


def load_metadata(lp):
    files_path = []
    for root, directories, files in os.walk(lp):
        for file in files:
            if '.hybrid' not in file:
                raw = file.split('.')
                testset = raw[0]
                lp = raw[-1]
                system = '.'.join(raw[1:-1])
                files_path.append((os.path.join(root, file), testset, lp, system))
    return files_path


def load_metadata_20(lp):
    files_path = []
    for root, directories, files in os.walk(lp):
        for file in files:
            if '.hybrid' not in file:
                raw = file.split('.')
                testset = raw[0]
                lp = raw[1]
                system = '.'.join(raw[2:-1])
                files_path.append((os.path.join(root, file), testset, lp, system))
    return files_path


def metric_combination(a, b, alpha):
    return alpha[0] * np.array(a) + alpha[1] * np.array(b)


def df_append(metric, num_samples, lp, testset, system, score):
    return pd.DataFrame({'metric': [metric] * num_samples,
                         'lp': [lp] * num_samples,
                         'testset': [testset] * num_samples,
                         'system': [system] * num_samples,
                         'sid': [_ for _ in range(1, num_samples + 1)],
                         'score': score,
                         })


def pearson(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]

    return str('{0:.{1}f}'.format(pearson_corr, 6))


def spearman(preds, labels):
    spearman_corr = spearmanr(preds, labels)[0]

    return str('{0:.{1}f}'.format(spearman_corr, 6))


def kendall(preds, labels):
    kendall_corr = kendalltau(preds, labels)[0]
    return str('{0:.{1}f}'.format(kendall_corr, 6))


from collections import defaultdict


def output_mt_correlation_17(lp_set, eval_metric, scores, f="Dataset/WMT17/DA-seglevel.csv", corr='pearson'):
    lines = [line.rstrip('\n') for line in open(f)]
    lines.pop(0)
    manual = {}
    for l in lines:
        l = l.replace("nmt-smt-hybrid", "nmt-smt-hybr")
        c = l.split()

        lp, data, system, sid, score = c[0], c[1], c[2], c[3], c[4]
        c = system.split("+")
        system = c[0]

        if lp not in manual:
            manual[lp] = {}
        if system not in manual[lp]:
            manual[lp][system + "::" + sid] = score

    correlations = defaultdict(list)

    submissions = [scores]

    for s in submissions:
        files = glob.glob(s)
        for f in files:
            correlations['setting'].append(f)
            missing = 0
            met_names = {}
            lms = {}
            lsm = {}

            lines = [line.rstrip('\n') for line in open(f)]
            for l in lines:
                l = l.replace("nmt-smt-hybrid", "nmt-smt-hybr")
                if (l.find("hybrid") == -1) and (l.find("himl") == -1):
                    c = l.split()
                    if len(c) < 6:
                        missing = missing + 1
                    else:
                        metric, lp, data, system, sid, score = c[0], c[1], c[2], c[3], c[4], c[5]
                        system = system + "::" + sid

                        if lp not in lms:
                            lms[lp] = {}

                        if metric not in lms[lp]:
                            lms[lp][metric] = {}
                        if system not in lms[lp][metric]:
                            lms[lp][metric][system] = score
                        if lp not in lsm:
                            lsm[lp] = {}

                        if system not in lsm[lp]:
                            lsm[lp][system] = {}
                        if system not in lsm[lp][system]:
                            lsm[lp][system][metric] = score

            for lp in manual:
                if lp not in lp_set: continue
                for metric in lms[lp]:
                    allthere = True
                    for trans in manual[lp]:
                        if not trans in lms[lp][metric]:
                            allthere = False
                            print(lp + " " + metric + " " + trans + '\n')
                    if allthere:
                        if lp not in met_names:
                            met_names[lp] = {}
                    if metric not in met_names[lp]:
                        met_names[lp][metric] = 1
                    else:
                        print("segment mismatch " + lp + " " + metric)
            for lp in manual:
                if lp not in lp_set: continue

                s = "LP SYSTEM HUMAN"
                for metric in sorted(met_names[lp]):
                    s = s + " " + metric
                s = s + "\n"
                for system in manual[lp]:
                    s = s + lp + " " + system + " " + manual[lp][system]
                    for metric in sorted(met_names[lp]):
                        s = s + " " + lsm[lp][system][metric]
                    s = s + "\n"
                results = pd.read_csv(StringIO(s), sep=" ")

                if corr == 'pearson':
                    correlation = pearson(results['HUMAN'], results[eval_metric])
                if corr == 'kendall':
                    correlation = kendall(results['HUMAN'], results[eval_metric])
                if corr == 'spearman':
                    correlation = spearman(results['HUMAN'], results[eval_metric])
                correlations[lp].append(correlation)

    output_df = pd.DataFrame.from_dict(correlations)
    s = 'SEG-' + corr + ' ' + (' '.join(lp_set) + '\n')
    scores = list(output_df.iloc[0, 1:].to_numpy())
    # s = s + ' ' + ' '.join([str('{0:.{1}f}').format(i, 6) for i in scores])
    s = s + metric + ' ' + ' '.join(scores)

    print(s + '\n')


def output_MT_sys_level_correlation(scores_path, lp_set, eval_metric, f, corr):
    submissions = [scores_path]
    lines = [line.rstrip('\n') for line in open(f)]
    lines.pop(0)

    manual = {}

    for l in lines:
        l = l.replace("nmt-smt-hybrid", "nmt-smt-hybr")
        l = l.replace('HUMAN-A', 'Human-A.0')
        l = l.replace('HUMAN-B', 'Human-B.0')
        l = l.replace('HUMAN-C', 'Human-P.0')
        l = l.replace('HUMAN', 'Human-A.0')
        if '20' in f:
            c = l.split(',')
        else:
            c = l.split()

        if len(c) != 3:
            print("erorr in manual evaluation file")
            exit(1)

        lp = c[0]
        score = c[1]
        system = c[2]

        if lp not in manual:
            manual[lp] = {}
        if system not in manual[lp]:
            manual[lp][system] = score

    missing = 0

    met_names = {}
    lms = {}
    lsm = {}

    for s in submissions:
        files = glob.glob(s)
        # print(files)
        for f in files:

            lines = [line.rstrip('\n') for line in open(f)]

            for l in lines:
                l = l.replace("nmt-smt-hybrid", "nmt-smt-hybr")

                if (l.find("hybrid") == -1) and (l.find("himl") == -1):

                    c = l.split()
                    # print(len(c))
                    # print(c)
                    if ((len(c) != 5) and len(c) != 7) and (len(c) != 9):
                        missing = missing + 1

                    else:
                        metric, lp, data, system, score = c[0], c[1], c[2], c[3], c[4]

                        '''
                        if data not in ["newstest2017", "newstest2018"]:
                          print ("error with data set for metric: "+l)
                          exit(1)
                        '''
                        if lp not in lms:
                            # print(lp)
                            lms[lp] = {}
                        if metric not in lms[lp]:
                            lms[lp][metric] = {}
                        if system not in lms[lp][metric]:
                            if system == 'online-B.0' and lp == 'gu-en' and data in ['newstest2019']:
                                continue
                            lms[lp][metric][system] = score

                        if lp not in lsm:
                            lsm[lp] = {}
                        if system not in lsm[lp]:
                            lsm[lp][system] = {}
                        if system not in lsm[lp][system]:
                            lsm[lp][system][metric] = score

    # print(manual.keys())
    # print(lsm.keys())

    for lp in manual:
        if lp not in lp_set: continue
        for metric in lms[lp]:
            if sorted(lms[lp][metric]) == sorted(manual[lp]):
                if lp not in met_names:
                    met_names[lp] = {}
                if metric not in met_names[lp]:
                    met_names[lp][metric] = 1

            else:
                print("systems mismatch " + lp + " " + metric)
                print(sorted(lms[lp][metric]))
                print(sorted(manual[lp]))
    res_str = ""
    for lp in manual:
        if lp not in lp_set: continue
        l = lp.replace("-", "")
        s = "LP SYSTEM HUMAN"

        for metric in sorted(met_names[lp]):
            s = s + " " + metric

        s = s + "\n"
        for system in manual[lp]:
            s = s + lp + " " + system + " " + manual[lp][system]
            for metric in sorted(met_names[lp]):
                s = s + " " + lsm[lp][system][metric]
            s = s + "\n"
        results = pd.read_csv(StringIO(s), sep=" ")
        if corr == 'pearson':
            res_str = res_str + corr + '\t' + lp + "\t" + pearson(results['HUMAN'], results[eval_metric]) + "\n"
        if corr == 'spearman':
            res_str = res_str + corr + '\t' + lp + "\t" + spearman(results['HUMAN'], results[eval_metric]) + "\n"
        if corr == 'kendall':
            res_str = res_str + corr + '\t' + lp + "\t" + kendall(results['HUMAN'], results[eval_metric]) + "\n"
    return pd.read_csv(StringIO(res_str), sep="\t", header=None)


import gzip


def output_MT_seg_level_correlation(scores_path, lp_set, eval_metric, f):
    submissions = [scores_path]

    lines = [line.rstrip('\n') for line in open(f)]
    # print(lines)
    lines.pop(0)

    manual = {}

    for l in lines:
        l = l.replace("nmt-smt-hybrid", "nmt-smt-hybr")
        l = l.replace('.zh-en', '')
        l = l.replace('rug-kken-', 'rug_kken_')
        l = l.replace('talp-upc-2019-kken', 'talp_upc_2019_kken')
        l = l.replace('Frank-s-MT', 'Frank_s_MT')
        l = l.replace('DBMS-KU-KKEN', 'DBMS-KU_KKEN')
        l = l.replace('Facebook-FAIR.6937', 'Facebook_FAIR.6937')
        l = l.replace('Helsinki-NLP.6889', 'Helsinki_NLP.6889')
        l = l.replace('Ju-Saarland.6525', 'Ju_Saarland.6525')
        l = l.replace('aylien-mt-gu-en-multilingual.6826', 'aylien_mt_gu-en_multilingual.6826')

        if '20' in f:
            c = l.split(',')


        else:
            c = l.split()
        # c = l.split()

        # c = l.split()[:-1]

        if len(c) != 5:
            # print('ffffffffffffff: ',len(c))
            print("error in manual evaluation file")
            # print(len(c))
            # exit(1)

        lp = c[0]
        data = c[1]
        sid = c[2]
        better = c[3]
        worse = c[4]

        if lp not in manual:
            manual[lp] = {}
        if sid not in manual[lp]:
            manual[lp][sid] = {}
        if better not in manual[lp][sid]:
            manual[lp][sid][better] = {}
        if worse not in manual[lp][sid][better]:
            manual[lp][sid][better][worse] = 1
    # print(manual)

    missing = 0
    met_names = {}
    metrics = {}

    for s in submissions:
        files = glob.glob(s)
        # print(files)
        for f in files:
            # lines = [str(line, encoding='utf-8') for line in gzip.open(f)]
            lines = [line.rstrip('\n') for line in open(f)]

            for l in lines:
                l = l.replace("nmt-smt-hybrid", "nmt-smt-hybr")
                l = l.replace("Unsupervised.de-cs", 'Unsupervised')
                l = l.replace("Unsupervised.cs-de", 'Unsupervised')
                if (l.find("hybrid") == -1) and (l.find("himl") == -1):
                    c = l.split()

                    if len(c) < 6:
                        missing = missing + 1

                    else:
                        metric = c[0]
                        lp = c[1]
                        data = c[2]
                        system = c[3]
                        sid = c[4]
                        score = float(c[5])
                        '''
                        if data != "newstest2018":
                            continue
                        '''
                        if lp not in metrics:
                            metrics[lp] = {}
                        if metric not in metrics[lp]:
                            metrics[lp][metric] = {}
                        if sid not in metrics[lp][metric]:
                            metrics[lp][metric][sid] = {}
                        if system not in metrics[lp][metric][sid]:
                            metrics[lp][metric][sid][system] = score

    # print(lp_set)
    for lp in manual:
        # print(lp)
        if lp not in lp_set: continue
        if lp not in metrics:
            print(lp + " not in metrics")
            exit(1)
        # print('dfsdfsdf')
        # print(metrics)
        for metric in metrics[lp]:
            allthere = True
            for sid in manual[lp]:
                if not sid in metrics[lp][metric]:
                    allthere = False
                    print("A) Missing " + lp + " " + metric + " " + sid + " no scores at all for this metric and sid")
                else:
                    for s1 in manual[lp][sid]:
                        if not s1 in metrics[lp][metric][sid]:
                            allthere = False
                            print(
                                "B) Missing " + lp + " " + metric + " " + sid + " " + s1 + " no scores for this metric for sid and first  system")
                    for s2 in manual[lp][sid][s1]:
                        if not s2 in metrics[lp][metric][sid]:
                            allthere = False
                            print(
                                "C) Missing " + lp + " " + metric + " " + sid + " " + s1 + " " + s2 + " no scores for this metric for sid and second system")

            if allthere:
                if lp not in met_names:
                    met_names[lp] = {}
                if metric not in met_names[lp]:
                    met_names[lp][metric] = 1

    res_str = ""
    # print(met_names)
    for lp in manual:
        if lp not in lp_set: continue

        for metric in met_names[lp]:
            conc = 0
            disc = 0
            for sid in manual[lp]:
                s = s + lp + " " + sid + " "
                for better in manual[lp][sid]:
                    for worse in manual[lp][sid][better]:
                        if better not in metrics[lp][metric][sid]:
                            print("error " + lp + " " + metric + " " + better)
                        score1 = metrics[lp][metric][sid][better]
                        score2 = metrics[lp][metric][sid][worse]
                        if score1 > score2:
                            conc = conc + 1
                        else:
                            disc = disc + 1

            conc = float(conc)
            disc = float(disc)
            result = (conc - disc) / (conc + disc)
            res_str = res_str + metric + '\t' + lp + "\t" + '{0:.{1}f}'.format(result, 6) + "\n"
            # print(res_str)
    return pd.read_csv(StringIO(res_str), sep="\t", header=None)


def print_sys_level_correlation(output_path, metric, data, lp_set, f="DA-syslevel.csv", corr='spearman'):
    results = pd.concat(data, ignore_index=True)
    del results['sid']
    results = results.groupby(['metric', 'lp', 'testset', 'system']).mean()
    results = results.reset_index()
    results.to_csv(output_path + '.sys.score', sep='\t', index=False, header=False)
    outputs = output_MT_sys_level_correlation(output_path + '.sys.score', lp_set, metric, f, corr)
    s = 'SYS-' + corr + ' ' + (' '.join(lp_set) + '\n')
    s = s + metric + ' ' + ' '.join(
        [str('{0:.{1}f}'.format(outputs[(outputs[1] == lp)].values[0][-1], 6)) for lp in lp_set])
    # print(corr)
    print(s + '\n')


import shutil


def print_seg_level_correlation(output_path, metric, data, lp_set, f="RR-seglevel.csv"):
    results = pd.concat(data, ignore_index=True)
    results.to_csv(output_path + '.seg.score', sep='\t', index=False, header=False)
    with open(output_path + '.seg.score', 'rb') as f_in:
        with gzip.open(output_path + '.seg.score.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    outputs = output_MT_seg_level_correlation(output_path + '.seg.score', lp_set, metric, f)
    # s = ' '.join(lp_set) + '\n'
    s = 'SEG-kendall ' + (' '.join(lp_set) + '\n')
    s = s + metric + ' ' + ' '.join(
        [str('{0:.{1}f}'.format(outputs[(outputs[1] == lp)].values[0][-1], 6)) for lp in lp_set])
    print(s + '\n')







