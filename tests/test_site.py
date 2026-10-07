import pandas as pd

from gridforecast import site


def test_issues_reports_real_issue_time_and_missed_days():
    log = pd.DataFrame({'issue_date': ['2026-10-04', '2026-10-05', '2026-10-07', '2026-10-07'],
                        'issued_at': ['2026-10-04T09:51:24+00:00', '2026-10-05T10:30:43+00:00',
                                      '2026-10-07T11:04:39+00:00', '2026-10-07T11:04:39+00:00']})
    assert site._issues(log) == {'last_issue_date': '2026-10-07', 'last_issued_at': '2026-10-07T13:04',
                                 'missing_issue_dates': ['2026-10-06']}
