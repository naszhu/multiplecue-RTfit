# Understanding the Express Component and Warning Messages

## What is the Express Component?

The **mixture model** has two components that combine to explain reaction times:

### 1. **Express Component** (Fast Responses)
- **What it represents**: Very fast, automatic, or reflexive responses
- **Examples**: Quick guesses, impulsive responses, or responses that bypass normal decision-making
- **Distribution**: Normal distribution with mean `mu_exp` and standard deviation `sig_exp`
- **Probability**: `p_exp` (what fraction of responses are express responses)
- **Typical RT range**: Usually < 0.3 seconds

### 2. **LBA Component** (Deliberate Responses)
- **What it represents**: Normal decision-making through the Linear Ballistic Accumulator model
- **Examples**: Thoughtful, deliberate choices based on evidence accumulation
- **Distribution**: LBA distribution (starts at `t0`, accumulates evidence to threshold)
- **Probability**: `1 - p_exp` (what fraction of responses are deliberate)
- **Typical RT range**: Usually > 0.25 seconds

## The Mixture Model

The total RT distribution is a **weighted combination**:
```
Total PDF = (p_exp × Express PDF) + ((1 - p_exp) × LBA PDF)
```

When these two components are **well-separated** and both have **substantial weight**, you get a **bimodal distribution** (two peaks).

## Warning Messages Explained

### Warning 1: "Express component collapsed (p_exp ≈ 0)"

**What it means**: The optimizer found that express responses are essentially not needed to explain the data.

**For Condition 9**: 
- `ProbExp = 7.1e-16` (0.0000000000000007% - essentially zero)
- This means the model found that **all responses can be explained by the LBA component alone**
- The express component has collapsed to zero

**Why this happens**: 
- The optimization algorithm tries to find the best fit
- If the LBA component alone fits the data well, it sets `p_exp` to near zero
- This is a valid solution - it means your data doesn't show evidence of fast express responses

**Visual result**: The plot shows only one mode (unimodal) because the express component contributes nothing.

---

### Warning 2: "Express component too small/close to LBA mode"

**What it means**: Even though `p_exp > 0`, the express component won't create visible bimodality because:

#### "Too small"
- The express component has very low probability (e.g., `p_exp = 0.01` means only 1% of responses)
- When you mix 1% express + 99% LBA, the 1% component gets visually overwhelmed
- The express peak is too weak compared to the main LBA peak

#### "Too close"
- The express component's peak location (mean RT) is too close to the LBA component's peak
- **Separation threshold**: Components need to be at least 100ms apart to create visible bimodality
- If express mean = 0.2s and LBA mode = 0.27s, they're only 70ms apart → too close
- The two components overlap too much, creating a single merged peak instead of two separate peaks

**Example from Condition 10**:
- `ProbExp = 0.0105` (1.05% - very small)
- `MuExp = 0.200s` (express mean)
- `t0 = 0.265s` (LBA starts here)
- Separation = 65ms (< 100ms threshold) → **too close**
- Result: Even though both components exist, they merge into one visible peak

**Visual result**: The plot appears unimodal even though technically it's a mixture, because the two components overlap too much.

---

## When Will You See Bimodality?

Bimodality (two visible peaks) requires **ALL** of these conditions:

1. ✅ `p_exp` is substantial (> 0.05 or 5% of responses)
2. ✅ Express mean is well-separated from LBA mode (> 100ms apart)
3. ✅ Express peak density is at least 10% of the main peak density

**Example of good bimodality**:
- `p_exp = 0.15` (15% express responses)
- `mu_exp = 0.15s` (express mean)
- LBA mode = 0.35s
- Separation = 200ms (> 100ms) ✅
- Express peak is substantial ✅

---

## For Condition 9 Specifically

Looking at the parameters:
- `ProbExp = 7.1e-16` (essentially zero)
- This triggers **Warning 1**: "Express component collapsed"

**What this means**: 
- The optimizer found that express responses are not needed
- All responses can be explained by the LBA component
- The model is essentially a single-component (LBA-only) model
- This is mathematically valid - it means your data doesn't require a mixture model

**Is this a problem?**
- **No, if** your data truly doesn't have fast express responses
- **Yes, if** you theoretically expect express responses but the optimizer isn't finding them
  - In this case, you might need to:
    - Use different initial values for `p_exp` (start at 0.1-0.2)
    - Constrain `p_exp` to be > 0.01 (force the optimizer to find express responses)
    - Check if your data actually has a fast RT mode that needs explaining

