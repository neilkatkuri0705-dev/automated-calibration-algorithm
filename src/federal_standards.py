"""
federal_standards.py — real-world reference dimensions for calibration.

Values are the realistic measured dimensions used by the calibration method
(document spec), consistent with US federal standards:

  - Dashed lane marking cycle L = 3.0 m paint + 9.0 m gap = 12.2 m.
    (MUTCD broken lane line; 10 ft + 30 ft = 40 ft = 12.19 m -> 12.2 m.)
    This is the ONLY longitudinal reference and fixes the product fH.
  - Lane width = 3.7 m (12 ft Interstate lane; AASHTO/FHWA).  Lateral ref.
  - Vehicle width = 1.8 m (realistic passenger-car body width). Lateral ref
    for estimating camera height H from detected vehicle boxes. NOTE: this is
    the actual car body width, NOT the wider AASHTO 2.13 m design envelope,
    because YOLO boxes the real body.

f (focal length) cannot be determined from width measurements alone — it
cancels in the width projection. Width fixes H; the lane-marking cycle L fixes
fH. See calibration_inverse_depth.py.
"""

FT_TO_M = 0.3048

# Longitudinal reference (dashed lane marking cycle)
DASH_PAINT_M = 3.0
DASH_GAP_M = 9.0
DASH_CYCLE_M = DASH_PAINT_M + DASH_GAP_M     # L = 12.2 m

# Lateral references
LANE_WIDTH_M = 3.7
VEHICLE_WIDTH_M = 1.8                          # realistic car body width

CITATIONS = {
    "dash_cycle_L": "MUTCD broken lane line 10ft+30ft=40ft ~= 12.2 m (3.0+9.0)",
    "lane_width": "12 ft Interstate lane (AASHTO/FHWA) ~= 3.7 m",
    "vehicle_width": "realistic passenger-car body width 1.8 m",
}
