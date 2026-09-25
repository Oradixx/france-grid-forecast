-- Interpolation is a patch for one known gap in the source (the repeated hour of the autumn
-- daylight-saving change, last Sunday of October). Anywhere else a missing hour is a real
-- ingestion problem and must fail the build.
select hour_utc
from {{ ref('fct_grid_hourly') }}
where is_interpolated
  and not (month(hour_utc) = 10 and dayofweek(hour_utc) = 0 and day(hour_utc) > 24)
