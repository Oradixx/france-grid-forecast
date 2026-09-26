"""Command line entry point: python -m gridforecast <command>

    ingest [--full-history]   download new data into the lake
    backtest [--start DATE]   train before DATE, score after it (report in lake/reports)
    train                     train a candidate, apply the quality gate, promote it if it passes
    predict [--date DATE]     forecast the day after DATE (default: today) and log it
    evaluate                  score published forecasts against measured consumption
    site                      build the dashboard (site/ + data.json) into ./public

`dbt build` runs between ingest and the rest (see the workflows): the Python steps read the
dbt marts from the DuckDB warehouse.
"""
import argparse
import json
import logging
import sys

from . import config, features, forecast, ingest, model, site


def main(argv=None):
    ap = argparse.ArgumentParser(prog='gridforecast', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('ingest')
    p.add_argument('--full-history', action='store_true', help='re-download the whole history')
    p = sub.add_parser('backtest')
    p.add_argument('--start', default=config.BACKTEST_START)
    p.add_argument('--if-stale', action='store_true', help='only if the report is missing or from other model code')
    p = sub.add_parser('train')
    p.add_argument('--if-stale', action='store_true', help='only if there is no model trained by the current code')
    p = sub.add_parser('predict')
    p.add_argument('--date', help='issue date (YYYY-MM-DD), the forecast is for the next day')
    p.add_argument('--replace', action='store_true', help="overwrite that day's published forecast")
    sub.add_parser('evaluate')
    p = sub.add_parser('site')
    p.add_argument('--out', default=str(config.ROOT / 'public'))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

    if args.cmd == 'ingest':
        ingest.run(full_history=args.full_history)
        return 0

    if args.cmd == 'train' and args.if_stale and not model.is_stale():
        print('current model matches the code, no retraining')
        return 0
    if args.cmd == 'backtest' and args.if_stale:
        rep = config.REPORTS / 'backtest.json'
        if rep.exists() and json.loads(rep.read_text()).get('model_signature') == model.code_signature():
            print('backtest report matches the code, skipped')
            return 0

    df = features.load()
    if args.cmd == 'backtest':
        report, test = model.backtest(df, args.start)
        forecast.write_json(report, config.REPORTS / 'backtest.json')
        print(json.dumps(report['overall'], indent=2))
    elif args.cmd == 'train':
        m, meta = model.train_candidate(df)
        promoted = model.promote(m, meta)
        print(json.dumps({'version': meta['version'], 'promoted': promoted, 'holdout': meta['holdout']}, indent=2))
        if not promoted and not (config.MODELS / 'current.json').exists():
            return 1  # no usable model at all
    elif args.cmd == 'predict':
        out = forecast.predict(df, args.date)
        log = forecast.append(out, replace=args.replace)
        kept = log[log['issue_date'] == out['issue_date'].iloc[0]]
        if not kept['forecast_mw'].equals(out['forecast_mw']):
            print(f"forecast issued on {out['issue_date'].iloc[0]} already published: kept unchanged")
        else:
            print(out[['target_hour_utc', 'forecast_mw', 'rte_forecast_d1_mw', 'temperature_c']].to_string(index=False))
    elif args.cmd == 'evaluate':
        ev = forecast.evaluate(df)
        forecast.write_json(ev, config.REPORTS / 'live.json')
        print(json.dumps(ev['summary'], indent=2))
    elif args.cmd == 'site':
        site.build(df, args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
