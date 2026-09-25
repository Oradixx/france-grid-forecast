{% test no_hour_gaps(model, column_name, where_not_null) %}
-- Every hour between the first and the last one that has a value must be present:
-- a missing hour means a failed download, not a quiet night.
with present as (
    select {{ column_name }} as h from {{ model }} where {{ where_not_null }} is not null
),
expected as (
    select unnest(generate_series(min(h), max(h), interval 1 hour)) as h from present
)
select e.h as missing_hour
from expected e
left join present p using (h)
where p.h is null
{% endtest %}
