# Power Outage Integration for Home Assistant

Monitor planned Energa distribution shutdowns for your Home Assistant zones. The integration fetches outage announcements directly from Energa and exposes them as sensor entities with rich details that can be surfaced in dashboards and automations.

## Features

- Tracks Energa shutdown notices that overlap with your configured Home Assistant zones.
- Provides a sensor per zone reporting the number of planned outages.
- Exposes formatted outage descriptions plus `startDate` and `endDate` attributes for the earliest shutdown in each zone.
- Supports both config entry setup and legacy YAML-based platform loading.

## Requirements

- Home Assistant 2023.0 or newer.
- Working internet connection to reach Energa's public outage endpoint.

## Installation (HACS)

1. In HACS, choose **Custom repositories** and add this repository as type **Integration**.
2. Search for **Power Outage** in HACS and install it.
3. Restart Home Assistant.
4. Add the integration via **Settings → Devices & Services → Add Integration** and select **Power Outage**.

For manual installation, copy the `custom_components/power_outage` directory into your Home Assistant `custom_components` folder and restart Home Assistant.

## Dashboard Example

You can use a conditional Lovelace card to show shutdown information only when an outage is scheduled for your "Home" zone sensor:

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

## Attributes

Every `sensor.power_outages_*` entity exposes these key attributes:

- `shutdowns`: Human-readable sentences describing each planned outage affecting the zone.
- `startDate`: Start time of the earliest planned outage (ISO timestamp).
- `endDate`: End time of the shutdown that begins earliest (ISO timestamp).
- `latitude`, `longitude`, `radius`: Zone geometry copied from the Home Assistant zone configuration.

## Troubleshooting

- Ensure your Home Assistant zones are configured with accurate latitude, longitude, and radius values.
- If no data appears, confirm Energa has published outage information for your area or check the Home Assistant logs for connectivity issues.

## License

Distributed under the MIT License. See `LICENSE` for details.
