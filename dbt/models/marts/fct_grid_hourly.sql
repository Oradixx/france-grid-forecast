-- Hourly national grid data, the table the model and the dashboard read.
-- Consumption is published every 15 or 30 minutes depending on the period: the hourly value is
-- the mean of the published points, and n_points says how many there were.
--
-- On the autumn daylight-saving day RTE publishes the repeated local hour only once, so one UTC
-- hour has no data at all. The table is built on a complete hourly spine and a missing hour
-- between two measured ones is linearly interpolated (is_interpolated = true); the lags of the
-- model need a continuous series. A test checks that this only ever happens on those days.
with points as (
    select
        date_trunc('hour', ts_utc)       as hour_utc,
        avg(consumption_mw)              as consumption_mw,
        count(consumption_mw)            as n_points,
        avg(rte_forecast_d1_mw)          as rte_forecast_d1_mw,
        avg(nuclear_mw)                  as nuclear_mw,
        avg(wind_mw)                     as wind_mw,
        avg(solar_mw)                    as solar_mw,
        avg(hydro_mw)                    as hydro_mw,
        avg(gas_mw)                      as gas_mw,
        avg(coal_oil_mw)                 as coal_oil_mw,
        avg(bioenergy_mw)                as bioenergy_mw,
        avg(net_imports_mw)              as net_imports_mw,
        avg(co2_g_per_kwh)               as co2_g_per_kwh,
        min(nature_rank)                 as nature_rank
    from {{ ref('stg_eco2mix') }}
    group by 1
),

spine as (
    select unnest(generate_series(min(hour_utc), max(hour_utc), interval 1 hour)) as hour_utc
    from points
),

joined as (
    select s.hour_utc, p.* exclude (hour_utc),
           lag(p.consumption_mw) over w  as prev_mw,
           lead(p.consumption_mw) over w as next_mw,
           lag(p.rte_forecast_d1_mw) over w  as prev_rte,
           lead(p.rte_forecast_d1_mw) over w as next_rte
    from spine s
    left join points p using (hour_utc)
    window w as (order by s.hour_utc)
)

select
    j.hour_utc,
    coalesce(j.consumption_mw, (j.prev_mw + j.next_mw) / 2)       as consumption_mw,
    j.consumption_mw is null and j.prev_mw is not null and j.next_mw is not null as is_interpolated,
    coalesce(j.n_points, 0)                                        as n_points,
    coalesce(j.rte_forecast_d1_mw, (j.prev_rte + j.next_rte) / 2) as rte_forecast_d1_mw,
    j.nuclear_mw, j.wind_mw, j.solar_mw, j.hydro_mw, j.gas_mw, j.coal_oil_mw, j.bioenergy_mw,
    j.net_imports_mw, j.co2_g_per_kwh, j.nature_rank,
    t.temperature_c,
    t.is_forecast as temperature_is_forecast
from joined j
left join {{ ref('fct_temperature_hourly') }} t using (hour_utc)
