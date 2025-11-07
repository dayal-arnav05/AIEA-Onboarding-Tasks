# LTL2TL Startup Guide

## Overview

**LTL2TL** is a tool that generates timeline visualizations for Linear Temporal Logic (LTL) formulas. It helps validate software system specifications by transforming LTL formulas into visual timelines and regular expressions.

### Key Features
- **ltl2timeline**: Converts LTL formulas to graphical timeline images
- **ltl2regex**: Translates LTL formulas to regular expressions
- Uses SPOT for Büchi automata transformation
- Supports multiple output formats (PDF, PNG, SVG, LaTeX)

---

## Quick Start with Docker (Recommended)

### 1. Pull and Run the Docker Image

```bash
# Pull the image
docker pull runmingl/ltl

# Run the container
docker run -it runmingl/ltl

# Navigate to the tool directory
cd /opt/ltl-explainability/src
```

### 2. Run Your First Example

```bash
# Generate a timeline visualization
python3 main.py ltl2timeline 'G(p xor X p)' --filename 'example' --output_format 'png'
```

This creates a timeline showing that property `p` alternates at every time step.

### 3. Copy Output to Your Local Machine

```bash
# In a new terminal (outside Docker), find your container ID
docker ps

# Copy the generated image
docker cp <containerId>:/opt/ltl-explainability/src/example.gv.png .
```

---

## LTL Syntax Reference

LTL formulas use SPOT syntax:

| Operator | Syntax | Meaning |
|----------|--------|---------|
| Negation | `!p` | NOT p |
| Conjunction | `p & q` | p AND q |
| Disjunction | `p \| q` | p OR q |
| Implication | `p -> q` | p implies q |
| XOR | `p xor q` | p XOR q |
| Next | `X p` | p holds in the next state |
| Eventually | `F p` | p holds eventually (Future) |
| Globally | `G p` | p holds globally (always) |
| Until | `p U q` | p holds until q |

---

## Common Use Cases & Examples

### Example 1: Request-Response Pattern
**Specification**: "Every request must eventually receive a response"

```bash
python3 main.py ltl2timeline 'G(request -> F response)' \
    --filename 'request_response' --output_format 'png'
```

### Example 2: Mutual Exclusion
**Specification**: "Two processes never access critical section simultaneously"

```bash
python3 main.py ltl2timeline 'G(!(process1 & process2))' \
    --filename 'mutual_exclusion' --output_format 'pdf'
```

### Example 3: Alternating Behavior
**Specification**: "Property p alternates between true and false"

```bash
python3 main.py ltl2timeline 'G(p xor X p)' \
    --filename 'alternating' --output_format 'svg'
```

### Example 4: Convert to Regular Expression
**Extract the formal regex representation**

```bash
python3 main.py ltl2regex 'G(request -> F response)'
```

---

## Command-Line Interface

### ltl2timeline

```bash
python3 main.py ltl2timeline '<FORMULA>' [OPTIONS]

Options:
  --filename        Output filename (default: 'ltl')
  --output_format   Format: 'pdf', 'png', 'svg', 'latex' (default: 'pdf')
```

### ltl2regex

```bash
python3 main.py ltl2regex '<FORMULA>'
```

---

## Local Installation (Alternative)

For development or production use on Unix-like systems:

### Prerequisites
- Python 3.10+
- SPOT library
- Graphviz

### Install Dependencies

```bash
# Install SPOT
./configure --prefix ~/.local && make && sudo make install

# Install Graphviz (macOS)
brew install graphviz

# Install Python packages
pip install -r requirements.txt

# Initialize git submodules
git submodule init
git submodule update
```

---

## Example Query Walkthrough

Let's validate a communication protocol specification:

**Scenario**: "In an aircraft communication system, every request must eventually stop"

**Formula**: `G(aircraft_request -> F(!aircraft_request))`

**Interpretation**:
- `G(...)` - Globally (always true)
- `aircraft_request -> ...` - When a request occurs
- `F(!aircraft_request)` - Eventually the request will be false (stops)

**Generate Timeline**:

```bash
python3 main.py ltl2timeline 'G(aircraft_request -> F(!aircraft_request))' \
    --filename 'aircraft_protocol' --output_format 'png'
```

**Result**: A visual timeline showing valid execution traces where aircraft requests eventually terminate.

---

## Troubleshooting

### Issue: Docker container doesn't have the tool
**Solution**: Ensure you're using the correct image: `docker pull runmingl/ltl`

### Issue: Output file not found
**Solution**: Check the current directory. Files are generated where you run the command.

### Issue: Syntax error in formula
**Solution**: Ensure proper quoting and use SPOT syntax. Single quotes around the formula are required.

---

## Additional Resources

- **GitHub Repository**: https://github.com/runmingl/ltl-explainability
- **SPOT Documentation**: https://spot.lrde.epita.fr/
- **Paper**: Available in the repository for detailed methodology

---

## Quick Reference Card

```bash
# Basic timeline generation
python3 main.py ltl2timeline '<FORMULA>' --filename 'output' --output_format 'png'

# Convert to regex
python3 main.py ltl2regex '<FORMULA>'

# Run example script (if available)
sh run_ltl_example.sh

# Copy files from Docker
docker cp <containerId>:/opt/ltl-explainability/src/<filename> .
```

**Remember**: Always wrap your LTL formula in single quotes to prevent shell interpretation!

