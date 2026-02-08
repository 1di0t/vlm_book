---
---

# 13.2 구조화 출력 평가

OCR이 단순 텍스트가 아닌 JSON, 테이블 등 구조화된 형태로 출력할 때의 평가 방법을 다룬다. 필드 단위 정확도, 구조 매칭, 스키마 검증까지 포괄한다.

> **핵심 용어**
>
> | 용어 | 정의 |
> |------|------|
> | **JSON Accuracy** | JSON 출력의 전체 정확도. 필드별 정확도의 가중 평균 |
> | **Field-level Accuracy** | 개별 필드가 정답과 일치하는 비율 |
> | **Schema Validation** | JSON 구조가 사전 정의된 스키마에 부합하는지 검증 |
> | **Tree Edit Distance** | 트리 구조 간 최소 편집 거리. JSON 구조 비교에 활용 |
> | **Exact Match** | 필드 값이 정답과 완전히 동일한지 판별 |
> | **Fuzzy Match** | 유사도 기반 부분 일치 판별 (NED 등 활용) |
> | **Structural Similarity** | 키 구조의 일치도. 값과 무관하게 스키마 형태만 비교 |
> | **Nested Accuracy** | 중첩 JSON 구조에서 재귀적으로 계산한 필드 정확도 |

---

## 수학적 원리

### 필드별 정확도 (Field-level Accuracy)

JSON 출력이 $K$개의 필드로 구성될 때, 각 필드 $k$에 대해:

$$
\text{Acc}_k = \frac{|\{i : \hat{v}_{i,k} = v_{i,k}\}|}{N}
$$

- $v_{i,k}$: $i$번째 샘플의 $k$번째 필드 정답값
- $\hat{v}_{i,k}$: $i$번째 샘플의 $k$번째 필드 예측값
- $N$: 총 샘플 수

**전체 필드 정확도 (가중 평균):**

$$
\text{Acc}_{\text{total}} = \frac{\sum_{k=1}^{K} w_k \cdot \text{Acc}_k}{\sum_{k=1}^{K} w_k}
$$

### Fuzzy Field Matching

정확히 일치하지 않더라도 유사한 값을 인정하는 방식이다. 필드 $k$의 fuzzy 정확도:

$$
\text{FuzzyAcc}_k = \frac{1}{N} \sum_{i=1}^{N} \left(1 - \text{NED}(\hat{v}_{i,k},\; v_{i,k})\right)
$$

$\text{NED}$는 Normalized Edit Distance다. 완전 일치면 1, 완전 불일치면 0에 수렴한다.

**임계값 기반 Fuzzy Match:**

$$
\text{Match}(v, \hat{v}) = \begin{cases}
1 & \text{if } \text{NED}(v, \hat{v}) \leq \tau \\
0 & \text{otherwise}
\end{cases}
$$

일반적으로 $\tau = 0.1$ (90% 이상 유사하면 매칭)을 사용한다.

### Tree Edit Distance

JSON 구조를 트리로 변환한 뒤 트리 편집 거리를 계산한다. Zhang-Shasha 알고리즘이 대표적이다.

트리 $T_1$과 $T_2$에 대해:

$$
d(T_1, T_2) = \min_{\text{edit ops}} \sum_{\text{op} \in \text{ops}} \text{cost}(\text{op})
$$

연산 종류:
- **Relabel**: 노드 레이블 변경 (비용 = $c_r$)
- **Delete**: 노드 삭제 (비용 = $c_d$)
- **Insert**: 노드 삽입 (비용 = $c_i$)

**정규화된 트리 편집 거리:**

$$
\text{NTED}(T_1, T_2) = \frac{d(T_1, T_2)}{|T_1| + |T_2|}
$$

### 중첩 구조 재귀적 평가

JSON이 중첩 구조를 가질 때, 재귀적으로 각 레벨의 정확도를 계산한다.

깊이 $l$에서의 정확도:

$$
\text{Acc}^{(l)} = \frac{\text{correct fields at depth } l}{\text{total fields at depth } l}
$$

**깊이 가중 전체 정확도:**

$$
\text{Acc}_{\text{nested}} = \sum_{l=0}^{L} \alpha^l \cdot \text{Acc}^{(l)}
$$

$\alpha \in (0, 1)$은 깊이 감쇠 계수다. 상위 필드에 더 높은 가중치를 부여한다.

### 스키마 준수율

$$
\text{SchemaCompliance} = \frac{|\{i : \text{validate}(\hat{y}_i, \mathcal{S}) = \text{True}\}|}{N}
$$

$\mathcal{S}$는 JSON Schema, $\hat{y}_i$는 $i$번째 예측 JSON이다.

---

## 코드 구현

### 기본 필드 비교 함수

```python
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


def normalize_value(value: Any) -> str:
    """
    비교를 위해 값을 정규화한다.
    공백 제거, 소문자 변환, 숫자 형식 통일.
    """
    if value is None:
        return ""
    s = str(value).strip()
    # 연속 공백 제거
    s = re.sub(r'\s+', ' ', s)
    return s


def exact_match(pred: Any, gold: Any) -> bool:
    """정확히 일치하는지 판별한다."""
    return normalize_value(pred) == normalize_value(gold)


def fuzzy_match(pred: Any, gold: Any, threshold: float = 0.1) -> bool:
    """
    NED 기반 fuzzy match.
    threshold 이하의 NED이면 매칭으로 판별한다.
    """
    s1 = normalize_value(pred)
    s2 = normalize_value(gold)
    if s1 == s2:
        return True

    # 간소화된 NED 계산
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return True

    dist = _levenshtein(s1, s2)
    ned = dist / max_len
    return ned <= threshold


def _levenshtein(s1: str, s2: str) -> int:
    """Levenshtein distance 계산."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[n]


def numeric_match(pred: Any, gold: Any, tolerance: float = 0.0) -> bool:
    """
    숫자 필드 비교. tolerance는 허용 오차 비율.
    tolerance=0이면 정확 일치, tolerance=0.01이면 1% 오차 허용.
    """
    try:
        # 쉼표, 원, 공백 등 제거
        pred_clean = re.sub(r'[,\s원₩]', '', str(pred))
        gold_clean = re.sub(r'[,\s원₩]', '', str(gold))
        pred_num = float(pred_clean)
        gold_num = float(gold_clean)

        if gold_num == 0:
            return pred_num == 0

        error_rate = abs(pred_num - gold_num) / abs(gold_num)
        return error_rate <= tolerance
    except (ValueError, TypeError):
        return exact_match(pred, gold)


def date_match(pred: Any, gold: Any) -> bool:
    """
    날짜 필드 비교. 다양한 형식을 통일 후 비교한다.
    '2024-03-15', '2024.03.15', '2024년 3월 15일' 등을 동일 취급.
    """
    def parse_date(s: str) -> Optional[str]:
        s = str(s).strip()
        # YYYY-MM-DD, YYYY.MM.DD, YYYY/MM/DD
        m = re.search(r'(\d{4})[.\-/년\s]+(\d{1,2})[.\-/월\s]+(\d{1,2})', s)
        if m:
            return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        return None

    pred_date = parse_date(str(pred))
    gold_date = parse_date(str(gold))

    if pred_date and gold_date:
        return pred_date == gold_date
    return exact_match(pred, gold)
```

### JSON 구조 비교 평가기

```python
@dataclass
class FieldResult:
    """단일 필드의 평가 결과"""
    field_name: str
    field_type: str  # 'string', 'number', 'date', 'nested', 'array'
    gold_value: Any
    pred_value: Any
    exact_match: bool
    fuzzy_match: bool
    similarity: float  # 0~1
    depth: int = 0


@dataclass
class JSONEvalResult:
    """JSON 비교 평가 전체 결과"""
    field_results: List[FieldResult] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    extra_fields: List[str] = field(default_factory=list)
    schema_valid: bool = True

    @property
    def exact_accuracy(self) -> float:
        if not self.field_results:
            return 0.0
        return sum(1 for f in self.field_results if f.exact_match) / len(self.field_results)

    @property
    def fuzzy_accuracy(self) -> float:
        if not self.field_results:
            return 0.0
        return sum(1 for f in self.field_results if f.fuzzy_match) / len(self.field_results)

    @property
    def avg_similarity(self) -> float:
        if not self.field_results:
            return 0.0
        return sum(f.similarity for f in self.field_results) / len(self.field_results)

    @property
    def structural_accuracy(self) -> float:
        """키 구조 일치율: 누락/추가 필드를 고려한다."""
        total = len(self.field_results) + len(self.missing_fields) + len(self.extra_fields)
        if total == 0:
            return 1.0
        return len(self.field_results) / total


class StructuredOutputEvaluator:
    """
    JSON 구조화 출력을 평가하는 클래스.

    필드 타입별로 적절한 비교 함수를 자동 선택한다.
    - 숫자 필드: numeric_match
    - 날짜 필드: date_match
    - 문자열 필드: exact_match + fuzzy_match
    - 중첩 구조: 재귀적 비교
    """

    def __init__(
        self,
        field_types: Optional[Dict[str, str]] = None,
        field_weights: Optional[Dict[str, float]] = None,
        fuzzy_threshold: float = 0.1,
        numeric_tolerance: float = 0.0,
    ):
        """
        Args:
            field_types: 필드명 -> 타입 매핑 (예: {'amount': 'number', 'date': 'date'})
            field_weights: 필드명 -> 가중치 매핑
            fuzzy_threshold: fuzzy match 임계값
            numeric_tolerance: 숫자 비교 허용 오차
        """
        self.field_types = field_types or {}
        self.field_weights = field_weights or {}
        self.fuzzy_threshold = fuzzy_threshold
        self.numeric_tolerance = numeric_tolerance
        self.results: List[JSONEvalResult] = []

    def _infer_field_type(self, key: str, value: Any) -> str:
        """필드 타입을 추론한다."""
        if key in self.field_types:
            return self.field_types[key]
        if isinstance(value, dict):
            return 'nested'
        if isinstance(value, list):
            return 'array'
        if isinstance(value, (int, float)):
            return 'number'
        s = str(value)
        if re.search(r'\d{4}[.\-/년]\s*\d{1,2}[.\-/월]\s*\d{1,2}', s):
            return 'date'
        if re.match(r'^[\d,.\s]+원?$', s):
            return 'number'
        return 'string'

    def _compute_similarity(self, pred: Any, gold: Any) -> float:
        """두 값의 유사도를 0~1로 계산한다."""
        s1 = normalize_value(pred)
        s2 = normalize_value(gold)
        if s1 == s2:
            return 1.0
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        dist = _levenshtein(s1, s2)
        return 1.0 - (dist / max_len)

    def _compare_field(
        self,
        key: str,
        pred_value: Any,
        gold_value: Any,
        depth: int = 0
    ) -> List[FieldResult]:
        """
        단일 필드를 비교한다. 중첩 구조는 재귀적으로 처리.

        Returns:
            FieldResult 리스트 (중첩 구조의 경우 여러 개)
        """
        ftype = self._infer_field_type(key, gold_value)

        # 중첩 dict 처리
        if ftype == 'nested' and isinstance(gold_value, dict):
            results = []
            pred_dict = pred_value if isinstance(pred_value, dict) else {}
            for sub_key in gold_value:
                if sub_key in pred_dict:
                    sub_results = self._compare_field(
                        f"{key}.{sub_key}",
                        pred_dict[sub_key],
                        gold_value[sub_key],
                        depth=depth + 1
                    )
                    results.extend(sub_results)
                else:
                    results.append(FieldResult(
                        field_name=f"{key}.{sub_key}",
                        field_type=self._infer_field_type(sub_key, gold_value[sub_key]),
                        gold_value=gold_value[sub_key],
                        pred_value=None,
                        exact_match=False,
                        fuzzy_match=False,
                        similarity=0.0,
                        depth=depth + 1
                    ))
            return results

        # 배열 처리
        if ftype == 'array' and isinstance(gold_value, list):
            results = []
            pred_list = pred_value if isinstance(pred_value, list) else []
            for idx in range(max(len(gold_value), len(pred_list))):
                g = gold_value[idx] if idx < len(gold_value) else None
                p = pred_list[idx] if idx < len(pred_list) else None
                if g is not None and p is not None:
                    sub_results = self._compare_field(
                        f"{key}[{idx}]", p, g, depth=depth + 1
                    )
                    results.extend(sub_results)
                else:
                    results.append(FieldResult(
                        field_name=f"{key}[{idx}]",
                        field_type='string',
                        gold_value=g,
                        pred_value=p,
                        exact_match=False,
                        fuzzy_match=False,
                        similarity=0.0,
                        depth=depth + 1
                    ))
            return results

        # Leaf 필드 비교
        is_exact = False
        is_fuzzy = False

        if ftype == 'number':
            is_exact = numeric_match(pred_value, gold_value, tolerance=0.0)
            is_fuzzy = numeric_match(pred_value, gold_value, tolerance=self.numeric_tolerance)
        elif ftype == 'date':
            is_exact = date_match(pred_value, gold_value)
            is_fuzzy = is_exact
        else:
            is_exact = exact_match(pred_value, gold_value)
            is_fuzzy = fuzzy_match(pred_value, gold_value, self.fuzzy_threshold)

        similarity = self._compute_similarity(pred_value, gold_value)

        return [FieldResult(
            field_name=key,
            field_type=ftype,
            gold_value=gold_value,
            pred_value=pred_value,
            exact_match=is_exact,
            fuzzy_match=is_fuzzy,
            similarity=similarity,
            depth=depth
        )]

    def evaluate_single(
        self,
        prediction: Dict,
        ground_truth: Dict
    ) -> JSONEvalResult:
        """
        단일 JSON 쌍을 평가한다.

        Args:
            prediction: OCR 예측 JSON
            ground_truth: 정답 JSON

        Returns:
            JSONEvalResult
        """
        result = JSONEvalResult()

        # 공통 키 비교
        gold_keys = set(ground_truth.keys())
        pred_keys = set(prediction.keys())

        result.missing_fields = list(gold_keys - pred_keys)
        result.extra_fields = list(pred_keys - gold_keys)

        for key in gold_keys:
            if key in prediction:
                field_results = self._compare_field(
                    key, prediction[key], ground_truth[key]
                )
                result.field_results.extend(field_results)
            else:
                # 누락 필드 — 모든 하위 필드도 오류
                field_results = self._compare_field(
                    key, None, ground_truth[key]
                )
                for fr in field_results:
                    fr.exact_match = False
                    fr.fuzzy_match = False
                    fr.similarity = 0.0
                result.field_results.extend(field_results)

        self.results.append(result)
        return result

    def evaluate_batch(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict]
    ) -> List[JSONEvalResult]:
        """여러 샘플을 일괄 평가한다."""
        assert len(predictions) == len(ground_truths)
        return [
            self.evaluate_single(pred, gold)
            for pred, gold in zip(predictions, ground_truths)
        ]

    def compute_report(self) -> dict:
        """전체 평가 리포트를 생성한다."""
        if not self.results:
            return {}

        n = len(self.results)

        # 전체 수준 메트릭
        report = {
            'num_samples': n,
            'avg_exact_accuracy': sum(r.exact_accuracy for r in self.results) / n,
            'avg_fuzzy_accuracy': sum(r.fuzzy_accuracy for r in self.results) / n,
            'avg_similarity': sum(r.avg_similarity for r in self.results) / n,
            'avg_structural_accuracy': sum(r.structural_accuracy for r in self.results) / n,
            'schema_compliance_rate': sum(1 for r in self.results if r.schema_valid) / n,
        }

        # 필드별 정확도
        field_stats = defaultdict(lambda: {'exact': 0, 'fuzzy': 0, 'total': 0, 'sim_sum': 0.0})
        for result in self.results:
            for fr in result.field_results:
                name = fr.field_name
                field_stats[name]['total'] += 1
                field_stats[name]['exact'] += int(fr.exact_match)
                field_stats[name]['fuzzy'] += int(fr.fuzzy_match)
                field_stats[name]['sim_sum'] += fr.similarity

        report['field_accuracy'] = {}
        for name, stats in field_stats.items():
            t = stats['total']
            report['field_accuracy'][name] = {
                'exact_accuracy': stats['exact'] / t,
                'fuzzy_accuracy': stats['fuzzy'] / t,
                'avg_similarity': stats['sim_sum'] / t,
                'count': t
            }

        return report

    def print_report(self):
        """평가 결과를 출력한다."""
        report = self.compute_report()
        print("=" * 65)
        print("구조화 출력 평가 리포트")
        print("=" * 65)
        print(f"  총 샘플 수:            {report['num_samples']}")
        print(f"  평균 Exact Accuracy:   {report['avg_exact_accuracy']:.4f}")
        print(f"  평균 Fuzzy Accuracy:   {report['avg_fuzzy_accuracy']:.4f}")
        print(f"  평균 Similarity:       {report['avg_similarity']:.4f}")
        print(f"  평균 구조 정확도:      {report['avg_structural_accuracy']:.4f}")
        print(f"  스키마 준수율:         {report['schema_compliance_rate']:.4f}")
        print()
        print("  필드별 Exact Accuracy:")
        for name, stats in report['field_accuracy'].items():
            print(f"    {name:30s} : {stats['exact_accuracy']:.4f} (n={stats['count']})")
        print("=" * 65)
```

### JSON Schema Validator

```python
class JSONSchemaValidator:
    """
    JSON 출력이 사전 정의된 스키마에 부합하는지 검증한다.
    경량 구현으로, jsonschema 라이브러리 없이 동작한다.

    스키마 형식:
    {
        "field_name": {
            "type": "string" | "number" | "date" | "object" | "array",
            "required": True | False,
            "pattern": "regex_pattern" (선택),
            "min_length": int (선택),
            "max_length": int (선택),
            "children": {...} (object 타입일 때),
            "items": {...} (array 타입일 때)
        }
    }
    """

    def __init__(self, schema: Dict):
        self.schema = schema

    def validate(self, data: Dict) -> Tuple[bool, List[str]]:
        """
        JSON 데이터를 스키마에 대해 검증한다.

        Returns:
            (is_valid, errors): 검증 결과와 오류 목록
        """
        errors = []
        self._validate_recursive(data, self.schema, "", errors)
        return len(errors) == 0, errors

    def _validate_recursive(
        self,
        data: Any,
        schema: Dict,
        path: str,
        errors: List[str]
    ):
        """재귀적으로 스키마 검증을 수행한다."""
        if not isinstance(schema, dict):
            return

        for field_name, field_spec in schema.items():
            current_path = f"{path}.{field_name}" if path else field_name

            # 필수 필드 체크
            if field_spec.get('required', False):
                if not isinstance(data, dict) or field_name not in data:
                    errors.append(f"필수 필드 누락: {current_path}")
                    continue

            if not isinstance(data, dict) or field_name not in data:
                continue

            value = data[field_name]
            expected_type = field_spec.get('type', 'string')

            # 타입 검증
            if expected_type == 'string' and not isinstance(value, str):
                errors.append(f"타입 불일치: {current_path} (기대: string, 실제: {type(value).__name__})")
            elif expected_type == 'number':
                if not isinstance(value, (int, float)):
                    # 문자열이지만 숫자로 파싱 가능한지 체크
                    try:
                        float(re.sub(r'[,\s원₩]', '', str(value)))
                    except ValueError:
                        errors.append(f"타입 불일치: {current_path} (기대: number, 실제: {type(value).__name__})")
            elif expected_type == 'object' and not isinstance(value, dict):
                errors.append(f"타입 불일치: {current_path} (기대: object, 실제: {type(value).__name__})")
            elif expected_type == 'array' and not isinstance(value, list):
                errors.append(f"타입 불일치: {current_path} (기대: array, 실제: {type(value).__name__})")

            # 패턴 검증
            if 'pattern' in field_spec and isinstance(value, str):
                if not re.match(field_spec['pattern'], value):
                    errors.append(f"패턴 불일치: {current_path} (패턴: {field_spec['pattern']}, 값: {value})")

            # 길이 검증
            if 'min_length' in field_spec and isinstance(value, str):
                if len(value) < field_spec['min_length']:
                    errors.append(f"최소 길이 미달: {current_path} ({len(value)} < {field_spec['min_length']})")
            if 'max_length' in field_spec and isinstance(value, str):
                if len(value) > field_spec['max_length']:
                    errors.append(f"최대 길이 초과: {current_path} ({len(value)} > {field_spec['max_length']})")

            # 중첩 구조 재귀
            if expected_type == 'object' and 'children' in field_spec and isinstance(value, dict):
                self._validate_recursive(value, field_spec['children'], current_path, errors)

            # 배열 아이템 검증
            if expected_type == 'array' and 'items' in field_spec and isinstance(value, list):
                for idx, item in enumerate(value):
                    item_path = f"{current_path}[{idx}]"
                    if isinstance(field_spec['items'], dict) and 'children' in field_spec['items']:
                        self._validate_recursive(
                            item, field_spec['items']['children'], item_path, errors
                        )

    def compliance_rate(self, data_list: List[Dict]) -> float:
        """여러 JSON에 대한 스키마 준수율을 계산한다."""
        if not data_list:
            return 0.0
        valid_count = sum(1 for d in data_list if self.validate(d)[0])
        return valid_count / len(data_list)
```

### Tree Edit Distance (간소화 버전)

```python
def json_to_tree(data: Any, key: str = "root") -> dict:
    """
    JSON 데이터를 트리 노드로 변환한다.

    Returns:
        {'label': str, 'children': list}
    """
    node = {'label': key, 'children': []}

    if isinstance(data, dict):
        for k, v in sorted(data.items()):
            child = json_to_tree(v, k)
            node['children'].append(child)
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            child = json_to_tree(item, f"[{idx}]")
            node['children'].append(child)
    else:
        # Leaf 노드: 값을 label에 포함
        node['label'] = f"{key}={data}"

    return node


def tree_size(node: dict) -> int:
    """트리의 노드 수를 세한다."""
    return 1 + sum(tree_size(c) for c in node.get('children', []))


def simple_tree_edit_distance(t1: dict, t2: dict) -> int:
    """
    간소화된 트리 편집 거리.
    전체 Zhang-Shasha보다 단순하지만 JSON 비교에 실용적이다.
    """
    if t1['label'] == t2['label']:
        cost = 0
    else:
        cost = 1

    c1 = t1.get('children', [])
    c2 = t2.get('children', [])

    # 자식 노드 매칭 (이름 기반)
    c1_map = {c['label'].split('=')[0]: c for c in c1}
    c2_map = {c['label'].split('=')[0]: c for c in c2}

    common_keys = set(c1_map.keys()) & set(c2_map.keys())
    only_c1 = set(c1_map.keys()) - common_keys
    only_c2 = set(c2_map.keys()) - common_keys

    for key in common_keys:
        cost += simple_tree_edit_distance(c1_map[key], c2_map[key])

    # 삭제/삽입 비용: 서브트리 크기
    for key in only_c1:
        cost += tree_size(c1_map[key])
    for key in only_c2:
        cost += tree_size(c2_map[key])

    return cost


def normalized_tree_edit_distance(json1: dict, json2: dict) -> float:
    """정규화된 트리 편집 거리를 계산한다."""
    t1 = json_to_tree(json1)
    t2 = json_to_tree(json2)
    dist = simple_tree_edit_distance(t1, t2)
    total_size = tree_size(t1) + tree_size(t2)
    if total_size == 0:
        return 0.0
    return dist / total_size
```

### 사용 예시

```python
if __name__ == "__main__":
    # 스키마 정의 — 보험 청구서 OCR 결과
    schema = {
        "patient_name": {"type": "string", "required": True, "min_length": 2},
        "diagnosis_code": {
            "type": "string", "required": True,
            "pattern": r"^[A-Z]\d{2,4}(\.\d{1,2})?$"
        },
        "total_amount": {"type": "number", "required": True},
        "admission_date": {"type": "string", "required": True},
        "discharge_date": {"type": "string", "required": True},
        "procedures": {
            "type": "array", "required": False,
            "items": {
                "children": {
                    "code": {"type": "string", "required": True},
                    "name": {"type": "string", "required": True},
                    "cost": {"type": "number", "required": True}
                }
            }
        }
    }

    # 정답
    ground_truth = {
        "patient_name": "김철수",
        "diagnosis_code": "K35.80",
        "total_amount": 1500000,
        "admission_date": "2024-03-10",
        "discharge_date": "2024-03-15",
        "procedures": [
            {"code": "Q2861", "name": "충수절제술", "cost": 800000},
            {"code": "N0501", "name": "전신마취", "cost": 200000}
        ]
    }

    # OCR 예측 (오류 포함)
    prediction = {
        "patient_name": "김철수",
        "diagnosis_code": "K35.8O",  # O(알파벳) vs 0(숫자) 오인식
        "total_amount": "1,500,000",  # 문자열로 출력됨
        "admission_date": "2024.03.10",  # 형식 다름
        "discharge_date": "2024-03-15",
        "procedures": [
            {"code": "Q2861", "name": "충수절제술", "cost": 800000},
            {"code": "NO501", "name": "전신마취", "cost": 200000}  # N0 -> NO
        ]
    }

    # 스키마 검증
    validator = JSONSchemaValidator(schema)
    is_valid, errors = validator.validate(prediction)
    print(f"스키마 유효: {is_valid}")
    for err in errors:
        print(f"  - {err}")

    # 구조화 출력 평가
    evaluator = StructuredOutputEvaluator(
        field_types={
            'total_amount': 'number',
            'admission_date': 'date',
            'discharge_date': 'date',
        },
        fuzzy_threshold=0.15
    )

    result = evaluator.evaluate_single(prediction, ground_truth)
    print(f"\nExact Accuracy: {result.exact_accuracy:.4f}")
    print(f"Fuzzy Accuracy: {result.fuzzy_accuracy:.4f}")
    print(f"Avg Similarity: {result.avg_similarity:.4f}")
    print(f"Structural Accuracy: {result.structural_accuracy:.4f}")

    # 필드별 상세
    print("\n필드별 결과:")
    for fr in result.field_results:
        status = "OK" if fr.exact_match else ("FUZZY" if fr.fuzzy_match else "FAIL")
        print(f"  [{status:5s}] {fr.field_name:30s} "
              f"gold={fr.gold_value!s:20s} pred={fr.pred_value!s:20s} "
              f"sim={fr.similarity:.3f}")

    # Tree Edit Distance
    ted = normalized_tree_edit_distance(prediction, ground_truth)
    print(f"\nNormalized Tree Edit Distance: {ted:.4f}")
```

---

## 구조화 출력 평가 시 주의사항

| 주의점 | 설명 |
|--------|------|
| **타입 일관성** | OCR이 숫자를 문자열로 출력하는 경우가 빈번하다. 타입 정규화 후 비교해야 한다 |
| **날짜 형식** | 다양한 날짜 형식을 파싱 후 통일 포맷으로 비교해야 공정하다 |
| **중첩 깊이** | 깊은 중첩 구조에서 하위 필드 하나 오류가 상위 구조 전체 오류로 전파되지 않게 주의 |
| **배열 순서** | 배열 내 아이템 순서가 다를 수 있다. 순서 무관 비교가 필요한 경우 별도 처리 |
| **Null 처리** | OCR이 필드를 인식하지 못하면 null이나 빈 문자열이 온다. 누락과 빈값을 구분해야 한다 |
| **부분 인식** | 필드의 일부만 인식된 경우 Exact Match는 실패하지만 Fuzzy Match로 부분 점수 부여 가능 |

---

## 용어 체크리스트

학습 후 아래 용어를 설명할 수 있는지 점검하라.

- [ ] **Field-level Accuracy**: 필드 단위로 평가하는 이유와 한계를 설명할 수 있는가?
- [ ] **JSON Accuracy**: Exact Match vs Fuzzy Match를 언제 쓰는지 구분할 수 있는가?
- [ ] **Schema Validation**: 스키마 검증이 왜 정확도 평가와 별도로 필요한지 아는가?
- [ ] **Tree Edit Distance**: JSON을 트리로 변환하는 과정을 설명할 수 있는가?
- [ ] **Normalized TED**: 정규화 분모가 왜 $|T_1| + |T_2|$인지 아는가?
- [ ] **Structural Accuracy**: 값 정확도와 구조 정확도의 차이를 아는가?
- [ ] **Fuzzy Threshold**: 임계값 설정이 평가 결과에 미치는 영향을 이해하는가?
- [ ] **Nested Accuracy**: 깊이 가중치 $\alpha$가 필요한 이유를 설명할 수 있는가?
