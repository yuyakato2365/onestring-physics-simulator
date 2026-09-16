# OptCuts × OneString: 固定Grid直交Seam融合

## 目的

従来の試作では、OptCutsが出した自由曲線Seamを、既に作られたM2D regular gridへ局所的にsnapしていた。この順序では、自由曲線を格子へ量子化する際にzig-zag、Seam端点の不整合、UV chartを跨ぐquad、T3D invalid tileが生じやすい。

本実装では役割を分ける。

- **OptCuts**: distortion-awareに「どこを切りたいか」を提案する。
- **OneString**: 実際に製作可能なSeam空間へ投影する。

## 固定する最小単位

新しい自動Grid sizeは導入しない。既存UIの `Tile size` をそのまま最小Grid単位 `h` とする。

したがって最終Seamの全edge長は基本的に `h` の整数倍で、Seamの端点・junction・折れ点はM2D grid vertex上にある。

## 共通直交軸

OptCutsのrobust seam graphから全Seam segmentを抽出し、長さ重み付きの主方向を求める。その主方向がUVの `u` 軸になるようUV全体をrigid rotationする。

これは2D剛体回転なのでOptCutsのJacobian singular valuesやSymmetric Dirichlet distortionを変えない。変わるのはGridとの相対方向だけである。

最終Seamはこの `u` 軸と、それに垂直な `v` 軸の2方向に限定する。

## Seam graphの圧縮

OptCutsの細かなedgeを1本ずつgridへsnapしない。

1. OptCuts seam graphを構築する。
2. degree != 2 の頂点をendpoint / junctionとして保存する。
3. degree = 2 の中間頂点列を1本のchainとして圧縮する。
4. 各chainを同じendpoint/junction間のrectilinear pathへ置換する。

これにより自由曲線の細かな揺れをそのまま階段状Seamへ変換しない。

## Rectilinear path

各chainについて、優先順位は次の通り。

1. 同一直線上ならstraight path 1本。
2. 可能ならhorizontal→vertical または vertical→horizontal のL字path。
3. Omega croppingのためL字が存在しない場合のみ、強いturn penaltyを持つgrid-only shortest path。

候補costは概念的に、

`path length + OptCuts source chainからの距離 + turn penalty`

で評価する。

fallbackでも全edgeはgridのu/v方向だけで、斜めedgeは作らない。

## Topology cut

Seam幅を作るためにcellを削除しない。選ばれたgrid edge上でincident face sectorごとにvertex IDをduplicateし、座標は同じままtopologyだけ切る。

したがって数値上のSeam gapは0である。

## 安全性

OptCuts UV chartを跨ぐquadを無理に復元しない。さらに既存の以下のguardを残す。

- M2D manifold guard
- inverse-map後にduplicate/degenerateになるquadの除外
- T3D前のK3D `validate_top_quad` preflight

理想状態ではこれらの除外数は0になるべきであり、除外が多い場合はrectilinear projectionとchart topologyの整合がまだ不十分だと判断する。

## 実行ログ

新しい経路では次を確認する。

- `[OPTCUTS-AXIS]`: OptCuts seam主方向へUVをrigid rotationした角度
- `[OPTCUTS-RECT]`: chain数、rectilinear path数、fallback数、cut edge数、invalid quad数、最小単位
- `[OPTCUTS-MANIFOLD]`: manifold guard結果
- `[OPTCUTS-K3D-PREFLIGHT]`: T3D直前のinvalid top数

特に `fallback=0`, `invalid_quads_removed=0`, `nonmanifold_edges_after=0`, `invalid_tops=0` が望ましい。
