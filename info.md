## Power Outage Integration

Monitor Energa distribution shutdowns that affect your Home Assistant zones. The integration exposes a sensor per zone with the number of planned outages plus detailed attributes for each shutdown, including the earliest `startDate` and `endDate` detected in the zone.

### Lovelace Example

```yaml
type: conditional
conditions:
  - condition: numeric_state
    entity: sensor.power_outages_home
    above: 0
card:
  type: markdown
  content: |-
    Planowane wyłączenie prądu:
    {{ state_attr('sensor.power_outages_home', 'shutdowns') }}
```

Enable the integration through **Settings → Devices & Services** after installing via HACS or by copying the `custom_components/power_outage` folder into your Home Assistant configuration directory.
