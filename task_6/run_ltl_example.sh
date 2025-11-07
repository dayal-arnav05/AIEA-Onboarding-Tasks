#!/bin/sh
# LTL2TL Example Run Script
# This script demonstrates the LTL2TL tool with example queries

echo "=========================================="
echo "LTL2TL Tool - Example Demonstrations"
echo "=========================================="
echo ""

# Check if running in Docker
if [ -d "/opt/ltl-explainability" ]; then
    cd /opt/ltl-explainability/src
else
    echo "Error: This script is designed to run in the ltl2timeline Docker container"
    echo "Please run: docker run -it runmingl/ltl"
    exit 1
fi

# Example 1: Basic alternating pattern
echo "Example 1: Alternating Pattern"
echo "Formula: G(p xor X p)"
echo "Description: Property p alternates at every time step"
echo ""
python3 main.py ltl2timeline 'G(p xor X p)' --filename 'example1_alternating' --output_format 'png'
echo "✓ Generated: example1_alternating.gv.png"
echo ""

# Example 2: Request-Response pattern
echo "Example 2: Request-Response Pattern"
echo "Formula: G(request -> F response)"
echo "Description: Every request is eventually followed by a response"
echo ""
python3 main.py ltl2timeline 'G(request -> F response)' --filename 'example2_request_response' --output_format 'png'
echo "✓ Generated: example2_request_response.gv.png"
echo ""

# Example 3: Aircraft Communication Protocol
echo "Example 3: Aircraft Communication Protocol"
echo "Formula: G(aircraft_request -> F(!aircraft_request))"
echo "Description: Every aircraft request eventually ends"
echo ""
python3 main.py ltl2timeline 'G(aircraft_request -> F(!aircraft_request))' --filename 'example3_aircraft' --output_format 'png'
echo "✓ Generated: example3_aircraft.gv.png"
echo ""

# Example 4: Safety property
echo "Example 4: Safety Property"
echo "Formula: G(!error | X(safe))"
echo "Description: If not in error state, next state must be safe"
echo ""
python3 main.py ltl2timeline 'G(!error | X(safe))' --filename 'example4_safety' --output_format 'png'
echo "✓ Generated: example4_safety.gv.png"
echo ""

# Example 5: Convert to regex
echo "Example 5: Converting LTL to Regular Expression"
echo "Formula: G(p xor X p)"
echo ""
python3 main.py ltl2regex 'G(p xor X p)'
echo ""

echo "=========================================="
echo "All examples completed!"
echo "=========================================="
echo ""
echo "To copy generated images to your local machine:"
echo "1. Find your container ID: docker ps"
echo "2. Copy files: docker cp <containerId>:/opt/ltl-explainability/src/example*.png ."
echo ""

