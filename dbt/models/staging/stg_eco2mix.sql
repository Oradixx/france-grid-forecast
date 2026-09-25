-- One row per 15-minute timestamp, from both éCO2mix datasets.
-- The raw files do not share one schema (RTE added columns over the years, and some columns are
-- text in the history but numbers in real time), so they are read by column name and every
-- measure is cast explicitly. When a timestamp is in both datasets, the most final version wins:
-- definitive > consolidated > real time.
--
-- Daylight saving time: RTE publishes one row per local clock time. On the spring-forward day the
-- non-existent local times 02:00-02:45 still get a row, stamped with the same UTC time as 03:00-03:45
-- (same consumption, but a different forecast value). Those phantom rows are dropped: a row is kept
-- only if its local date and time match its UTC timestamp.
with raw as (
    select * from read_parquet('{{ var("lake") }}/raw/eco2mix/*/*.parquet', union_by_name = true, filename = true)
),

typed as (
    select
        cast(date_heure as timestamptz) at time zone 'UTC'   as ts_utc,
        nature,
        case
            when nature ilike '%définitives%' then 1
            when nature ilike '%consolidées%' then 2
            else 3
        end                                                   as nature_rank,
        strptime(date || ' ' || heure, '%Y-%m-%d %H:%M')
            = (cast(date_heure as timestamptz) at time zone 'Europe/Paris') as local_time_matches,
        try_cast(consommation as double)                      as consumption_mw,
        try_cast(prevision_j1 as double)                      as rte_forecast_d1_mw,
        try_cast(nucleaire as double)                         as nuclear_mw,
        try_cast(eolien as double)                            as wind_mw,
        try_cast(solaire as double)                           as solar_mw,
        try_cast(hydraulique as double)                       as hydro_mw,
        try_cast(gaz as double)                               as gas_mw,
        try_cast(charbon as double) + try_cast(fioul as double) as coal_oil_mw,
        try_cast(bioenergies as double)                       as bioenergy_mw,
        try_cast(ech_physiques as double)                     as net_imports_mw,
        try_cast(taux_co2 as double)                          as co2_g_per_kwh
    from raw
    where date_heure is not null
)

select *
from typed
qualify row_number() over (partition by ts_utc order by nature_rank, local_time_matches desc) = 1
