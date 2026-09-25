-- Hourly temperature per city: observed values (archive) first, then for the hours the archive
-- does not cover yet, the latest forecast issued for them.
with archive as (
    select cast(time_utc as timestamptz) at time zone 'UTC' as hour_utc, city, temperature,
           'observed' as kind, 1 as rank_, null::date as issued
    from read_parquet('{{ var("lake") }}/raw/weather/archive/*.parquet')
),

forecast as (
    select cast(time_utc as timestamptz) at time zone 'UTC' as hour_utc, city, temperature,
           'forecast' as kind, 2 as rank_, cast(issued as date) as issued
    from read_parquet('{{ var("lake") }}/raw/weather/forecast/*.parquet')
)

select hour_utc, city, temperature, kind
from (select * from archive union all select * from forecast)
qualify row_number() over (partition by hour_utc, city order by rank_, issued desc) = 1
