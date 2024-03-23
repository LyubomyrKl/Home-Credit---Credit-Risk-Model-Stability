import scipy


def check_ttest_by_target(ids, base):
    filtered_targets = base.filter(base['case_id'].is_in(ids))['target']
    base_targets = base['target']

    filtered_targets_percent = filtered_targets.sum() / len(filtered_targets)
    base_targets_percent = base_targets.sum() / len(base)

    t_statistic, p_value = scipy.stats.ttest_ind(filtered_targets, base_targets)

    print("T-statistic:", t_statistic)
    print("P-value:", p_value)
    print("Unique targets percent:", filtered_targets_percent)
    print("Base targets percents:", base_targets_percent)

    return t_statistic, p_value, filtered_targets_percent, base_targets_percent
