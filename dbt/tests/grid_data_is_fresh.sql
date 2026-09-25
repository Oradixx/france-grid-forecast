-- The latest measured consumption must be less than 36 hours old, otherwise the forecast would
-- be built on stale lags: fail the run instead of publishing it.
{{ config(enabled = var('check_freshness')) }}
select max(hour_utc) as last_measured_hour
from {{ ref('fct_grid_hourly') }}
where consumption_mw is not null
having max(hour_utc) < (now() at time zone 'UTC') - interval 36 hour
