-- National temperature: population-weighted mean of the cities (see seeds/cities.csv).
-- Only hours where every city has a value, so the weights always mean the same thing.
select
    w.hour_utc,
    sum(w.temperature * c.weight) / sum(c.weight) as temperature_c,
    bool_or(w.kind = 'forecast')                  as is_forecast
from {{ ref('stg_weather') }} w
join {{ ref('cities') }} c using (city)
group by w.hour_utc
having count(*) = (select count(*) from {{ ref('cities') }})
