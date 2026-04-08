# 🤝 Contributing to ADAS Perception Simulation Stack

Thank you for your interest in contributing! This document explains how to work with the codebase, open pull requests, and adhere to the project's code standards.

---

## Table of Contents

1. [Getting Set Up](#1-getting-set-up)
2. [Branch Strategy](#2-branch-strategy)
3. [Commit Message Format](#3-commit-message-format)
4. [C++ Code Style](#4-c-code-style)
5. [Python Code Style](#5-python-code-style)
6. [Adding Tests](#6-adding-tests)
7. [Pull Request Checklist](#7-pull-request-checklist)
8. [Review Etiquette](#8-review-etiquette)

---

## 1. Getting Set Up

```bash
git clone https://github.com/RAVIVANAMA/adas_perception_simulation.git
cd adas_perception_simulation

# Build in Debug mode (enables assertions)
cmake -S . -B build/Debug -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build/Debug

# Run tests to confirm clean baseline
cd build/Debug && ctest --output-on-failure
```

---

## 2. Branch Strategy

| Branch prefix | Purpose | Example |
|---|---|---|
| `main` | Stable releases only | — |
| `feat/` | New feature or module | `feat/kf-2d-tracker` |
| `fix/` | Bug fix | `fix/aeb-false-positive` |
| `docs/` | Documentation changes | `docs/architecture-diagram` |
| `refactor/` | Non-behavioral restructuring | `refactor/extract-pid-class` |
| `test/` | New or improved tests | `test/acc-edge-cases` |
| `ci/` | CI/CD pipeline changes | `ci/add-clang-tidy` |

Do **not** commit directly to `main`. Always open a pull request.

---

## 3. Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

```
<type>(<scope>): <short imperative summary>

[optional body — explain what and why, not how]

[optional footer — e.g., Closes #12, BREAKING CHANGE: ...]
```

**Types:** `feat`, `fix`, `docs`, `refactor`, `test`, `build`, `ci`, `perf`, `style`, `chore`

**Examples:**

```
feat(aeb): add multi-object TTC aggregation
fix(sensor-fusion): prevent stale track accumulation on reset
docs(architecture): add EKF state-transition matrix diagram
test(acc): cover edge case when lead vehicle is stationary
```

---

## 4. C++ Code Style

### General Rules

- Standard: **C++17** — use modern features (`std::optional`, `if constexpr`, structured bindings, `std::string_view`)
- No raw `new` / `delete` — use `std::unique_ptr`, `std::shared_ptr`, or stack allocation
- No global mutable state (except `Logger` singleton, which is thread-safe)
- All functions < 60 lines where practical; extract helpers for clarity

### Naming

| Entity | Convention | Example |
|---|---|---|
| Types / Classes | `PascalCase` | `ObjectDetector` |
| Functions / Methods | `camelCase` | `computeTTC()` |
| Member variables | `snake_case_` with trailing `_` | `max_age_` |
| Local variables | `snake_case` | `ego_speed` |
| Constants / Enums | `UPPER_SNAKE_CASE` | `DEFAULT_HEADWAY` |
| Namespaces | `snake_case` | `adas::planning` |

### Include Order (clang-format: `IncludeBlocks: Regroup`)

```cpp
// 1. Corresponding header (if .cpp)
#include "planning/aeb_controller.hpp"

// 2. Other project headers
#include "common/logger.hpp"
#include "common/math_utils.hpp"

// 3. Third-party headers
#include <Eigen/Dense>

// 4. Standard library
#include <algorithm>
#include <cmath>
#include <vector>
```

### Formatting

- Indent: 4 spaces (no tabs)
- Brace style: Allman (opening brace on new line for functions, inline for control flow)
- Line length: 100 characters maximum
- Apply `clang-format` before committing:

```bash
find src include tests -name "*.cpp" -o -name "*.hpp" | \
  xargs clang-format --style=file -i
```

A `.clang-format` file enforcing these rules is in the repo root.

### Error Handling

- Use `LOG_ERROR` / `LOG_FATAL` from `logger.hpp` for unrecoverable failures.
- Return `bool` or `std::optional<T>` to signal recoverable errors; avoid exceptions in hot paths.

---

## 5. Python Code Style

- Follow **PEP 8** with a maximum line length of 99 characters.
- Use **type hints** on all public function signatures.
- Format with **black** and lint with **flake8** or **ruff**:

```bash
pip install black ruff
black python/ --line-length 99
ruff check python/
```

- Docstrings: Google style (Args / Returns / Raises sections).
- Keep each script runnable standalone (`if __name__ == "__main__": ...`).

---

## 6. Adding Tests

### C++ (Google Test)

1. Add a new file under `tests/` named `test_<component>.cpp`.
2. Register it in `tests/CMakeLists.txt` under `target_sources(adas_tests ...)`.
3. Each test function should:
   - Have a descriptive name: `TEST(AEBController, FullBrakeBelow2SecTTC)`.
   - Construct only the component under test — no global state.
   - Assert a single logical outcome per `TEST()`.

```cpp
TEST(ACCController, ThrottleWhenBelowSetSpeed) {
    ACCControllerConfig cfg;
    cfg.setSpeed = 30.0f;
    ACCController acc(cfg);

    auto out = acc.update({}, {0.0f, 0.0f, 0.0f}, 25.0f, 0.033f);
    EXPECT_GT(out.throttle, 0.0f);
    EXPECT_FLOAT_EQ(out.brake, 0.0f);
}
```

### Python

- Use **pytest**.
- Place tests under `python/tests/`.
- Name files `test_<module>.py` and functions `test_<behavior>`.

---

## 7. Pull Request Checklist

Before opening a PR, confirm:

- [ ] Branch is up-to-date with `main` (`git rebase origin/main`)
- [ ] Code compiles in both **Release** and **Debug** modes
- [ ] All existing tests pass (`ctest --output-on-failure`)
- [ ] New tests added for new functionality
- [ ] `clang-format` applied (C++) / `black` + `ruff` applied (Python)
- [ ] No new compiler warnings (`-Wall -Wextra`)
- [ ] PR title follows Conventional Commits format
- [ ] Description explains _what_ changed and _why_

---

## 8. Review Etiquette

- Be specific in review comments — link to lines, cite standards.
- Distinguish **blocking** issues from **suggestions**: prefix suggestions with `nit:`.
- Respond to all comments before requesting re-review.
- Maintainers aim to review PRs within 3 business days.

---

*Thank you for helping make this project better!*
