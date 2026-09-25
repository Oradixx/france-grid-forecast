-- Sanity check across sources: RTE's day-ahead forecast and the measured consumption describe
-- the same grid, so over a whole day they cannot differ by more than 15 %. A bigger gap means a
-- unit or timestamp problem (for example a timezone shift) in one of the two columns.
with daily as (
    select date_trunc('day', hour_utc) as day,
           sum(consumption_mw) as actual, sum(rte_forecast_d1_mw) as rte, count(*) as hours
    from {{ ref('fct_grid_hourly') }}
    where consumption_mw is not null and rte_forecast_d1_mw is not null
    group by 1
)
select * from daily
where hours = 24 and abs(rte - actual) / actual > 0.15
