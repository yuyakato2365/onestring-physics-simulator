# Native Grid-OptCuts — OneString用設計契約

## 目的

`optcuts_grid` は「通常のOptCutsを完走してからSeamをGridへ変形する後処理」ではない。
OptCuts自身の **cut candidate search（切断候補探索）** を改造し、最初から製作可能なGrid cutだけを探索空間に含める。

概念的には

\[
\min_{C,\Phi} E_{SD}(\Phi) + \lambda L(C)
\qquad\text{s.t.}\qquad C\in\mathcal C_{grid}
\]

を解く。

- `C`: cut / seam topology
- `Phi`: surface parameterization
- `E_SD`: Symmetric Dirichlet distortion
- `C_grid`: OneStringの固定直交Grid上で製作可能なcut集合

## 絶対に守る不変条件

1. **Grid制約はOptCutsの候補評価前に適用する。**
   - 完成後の自由SeamをGridへsnapしてはいけない。
   - Python continuationで自由OptCuts UVをGridへ押し込んではいけない。

2. **Grid spacing `h` はOneStringの `Tile size` と同一。**
   - OptCuts実行後にUVをarea-normalize / rescaleしてはいけない。

3. **許可されるSeam segment方向は、1つのglobal orthogonal frameの2軸のみ。**
   - H
   - V
   - H→V
   - V→H
   - 連続するsegmentによりより長いGrid polylineを構成してよい。

4. **一度採用されたSeam / junction頂点はGrid上で永久lockする。**
   - 後続cut候補が既存junctionを別Grid交点へ移動してはいけない。

5. **候補は実際のtopology cutをtrialしてから採点対象にする。**
   - Grid位置に置いた実cutでtriangle inversionが起きる候補はrejectする。

6. **native Grid-OptCuts出力をPythonで再parameterizeしない。**
   - rigid frame rotation/reflectionは可（距離・角度・hを変えない）。
   - scale / non-rigid warp / post-hoc seam snapは禁止。

7. **M2Dは同じ `h` / phaseのGridを使う。**
   - C++が実際に選んだOptCuts seamを、M2Dの一致するGrid edgeへzero-width topology cutとして転写する。
   - 座標を開かず、vertex IDだけ複製する。
   - 第2のSeamを追加してはいけない。

8. **official `optcuts` と `optcuts_grid` は完全分離する。**
   - `optcuts`: authors' OptCuts結果をそのまま利用。
   - `optcuts_grid`: native candidate-search modificationのみ。

## Native V2の処理

```text
S: target triangle surface
↓
OptCuts initial UV
↓
querySplit / computeLocalLDec
↓
通常のtopological split candidate
↓
Grid embedding candidate生成
    H / V / H→V / V→H
    spacing = h = Tile size
    global orientation / phase固定
↓
既存Grid junction lockを動かしていないか確認
↓
実際にtrial cut
↓
triangle inversionがあればreject
↓
OptCuts local SD improvement
  - Grid snap SD cost
  - seam length cost
で比較
↓
best feasible candidateをcutPath / splitEdgeOnBoundaryへ適用
↓
新しいSeam頂点とcopyをGrid lock
↓
通常のOptCuts UV optimization
  （lockされたSeam座標は動かない）
↓
次のGrid cut探索
↓
raw UV export（scale禁止）
↓
global positive-area UV overlap audit
↓
同じh/phaseのM2D Grid
↓
native OptCuts seamをzero-width topology cutとして転写
↓
M3D → K3D → T3D / K2D ...
```

## V2で明示的に禁止しているもの

- OptCuts merge
  - 現在のmergeはGrid lock topologyを保持する保証がないため、`optcuts_grid` ではsplit-only。
- OptCuts air-mesh scaffold
  - Seam両copyを同じfabrication Grid lineへ置くため、現行scaffoldの分離境界仮定と両立しない。
  - 代わりにlocal inversion check + final global positive-area UV overlap auditを必須化。
- Python post-hoc Grid fusion
- Python continuation optimizer
- arbitrary fallback zig-zag
- Seam crossing cellの「heal」や自由Seamを残した二重Seam

## Native V2の既知の制限

OptCutsのtopological candidateはまだ**入力surface triangle meshの既存edge**に沿う。
つまり、Grid lineと3D triangleの任意交点に新しいsurface vertexを挿入してcut候補を作るところまでは実装していない。

これは「Grid制約を後処理に戻す」という意味ではない。採用されるSeamはすべてGrid上にあるが、探索できるcut topologyの解像度がsource triangulationに制限される。

将来の拡張は、Grid line / current UV triangle intersectionからsurface上の新規cut vertexを挿入し、`C_grid` 自体をより高解像度にすること。

## 失敗時の扱い

エラーを回避するためにGrid制約を緩めてはいけない。

- Grid候補がない → candidate reject / explicit infeasible
- trial cutで反転 → candidate reject
- final local flip / degeneracy → stop
- global UV overlap → stop
- native seam endpointがM2D lattice上にない → stop
- native seamがH/Vでない → stop

この失敗は「後からSeamを修理する」トリガーではなく、Grid-constrained candidate searchを改善するための診断情報として扱う。
