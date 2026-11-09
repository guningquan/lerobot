# ALOHA ROS × LeRobot 日本語ガイド

[中文版](./README_CN.md)

コードのダウンロードからロボットでモデルを動かすまでの全手順：環境構築 → ロボット起動 → キャリブレーション → 遠隔操作 → データ収集 → 変換 → 学習 → デプロイ・実行 → HIL による追加データ収集 → スリープ・電源オフ。

対象環境：本リポジトリの `aloha2` ブランチ。ロボット PC は Ubuntu 20.04 + ROS Noetic + Interbotix。4 アーム構成の ALOHA（リーダーアーム（master）WidowX `wx250s` ×2、フォロワーアーム（puppet）ViperX `vx300s` ×2）+ USB カメラ 4 台（`cam_high`、`cam_low`、`cam_left_wrist`、`cam_right_wrist`）。

このマシンでのパス（別のマシンでは自分の環境に読み替えてください）：

| 用途 | パス |
| --- | --- |
| コード | `/mnt/server_ssd2/Programs_server/lerobot` |
| LeRobot データセット | `/home/robot/Dataset_and_Checkpoint/lerobot-dataset/` |
| ALOHA2 の生 hdf5 | `/home/robot/Dataset_and_Checkpoint/dataset/<task_name>/` |
| デプロイ用モデル | `/home/robot/Dataset_and_Checkpoint/checkpoints/` |

---

## 0. まずデータのルートを決める

データの作り方は 2 通りあります。**学習に使ったデータのルートに合わせて、デプロイ時のパラメータも必ず揃えてください。** 揃っていないと、モデルが受け取るグリッパ状態の意味が学習時と変わり、グリッパがおかしな動きをします。

| | ルート A：LeRobot で直接収録 | ルート B：ALOHA2 hdf5 + 変換 |
| --- | --- | --- |
| 収録スクリプト | `aloha_ros_record.py` または `lerobot-record` | act-triple-plus-aloha2 の `record_episodes_aloha2.py` |
| 出力 | LeRobot v3.0 データセット | `episode_N.hdf5`、これを act-triple の `convert_aloha_hdf5_to_lerobot_v30.py` で v3.0 に変換 |
| グリッパ状態 | 関節角を [0, 1] に正規化 | ALOHA2 の qpos と同じ正規化 |
| 追加の特徴量 | なし | `observation.velocity`、`observation.effort` |
| `aloha_ros_eval.py` | `--data_format lerobot`（デフォルト） | `--data_format aloha2` |
| `lerobot-rollout` | グリッパ関連の引数は付けない | `--robot.gripper_state_format=aloha2` |
| 既存の例 | apple_grasping（ACT、30 Hz） | packaging_simple_2（pi05、MolmoAct2、50 Hz） |

アクション（action）のグリッパの意味はどちらのルートでも同じで、正規化した開き具合（0 = 閉、1 = 開）です。アクションはどちらも**絶対**の関節目標値で、これが LeRobot のデフォルトです。本ガイドの学習コマンドはすべて絶対アクションを前提にしています。

---

## 1. コードのダウンロード

```bash
cd /mnt/server_ssd2/Programs_server
git clone git@github.com:guningquan/lerobot.git
cd lerobot
git checkout aloha2
git remote add upstream https://github.com/huggingface/lerobot.git   # 本家の更新を取り込むとき用
```

---

## 2. 環境構築

### 2.1 前提条件

- ROS Noetic と Interbotix のワークスペース `~/interbotix_ws_noetic` がビルド済みで、`aloha` パッケージ（`4arms_tactile_teleop.launch` を含む）が入っていること。
- [uv](https://docs.astral.sh/uv/) がインストール済みであること。

### 2.2 Python 環境（uv、Python 3.12）

```bash
cd /mnt/server_ssd2/Programs_server/lerobot
uv venv --python 3.12                       # .venv がすでにあれば不要
uv sync --locked --extra core_scripts --extra training --extra pi --extra molmoact2
uv pip install rospkg catkin_pkg netifaces h5py
```

- `pi` は pi0/pi0.5、`molmoact2` は MolmoAct2 の依存パッケージです。ACT だけを学習するなら不要です。
- 最後の行は Python 3.12 から `import rospy` するためのものです。これらは `uv.lock` に含まれていないので、**`uv sync` のたびに削除されます**。sync した後はこの行をもう一度実行してください（または `uv sync --inexact` を使う）。足りないと `ModuleNotFoundError: No module named 'rospkg'` になります。
- pi05 の学習では Hugging Face から `lerobot/pi05_base` と `google/paligemma-3b-pt-224` のトークナイザをダウンロードします。後者は先に Web ページで PaliGemma のライセンスに同意し、`hf auth login` しておく必要があります。デプロイだけなら不要です（checkpoint にトークナイザが含まれています）。

### 2.3 新しいターミナルごとの初期化

```bash
source ~/interbotix_ws_noetic/devel/setup.bash
source /mnt/server_ssd2/Programs_server/lerobot/.venv/bin/activate
cd /mnt/server_ssd2/Programs_server/lerobot
```

順番を逆にしないでください。`setup.bash` が設定する `PYTHONPATH` によって venv から rospy と interbotix が見えるようになります。`unset PYTHONPATH` もしないでください。起動直後に出る `/opt/ros/noetic/... SyntaxWarning: invalid escape sequence` は、ROS の Python 3.8 向けコードを 3.12 で読み込んだときの警告なので無視して構いません。

### 2.4 よく使う変数

以下のコマンドはこれらの変数を使います。**変数は今のターミナルでしか有効ではありません。** 新しいターミナルでは設定し直してください。変数が空だと `Couldn't parse '' into an int` のようなエラーになります。`~/aloha_env.sh` に保存して、毎回 `source ~/aloha_env.sh` するのがおすすめです。

```bash
FPS=50                                  # データセットの fps と同じにする：packaging 系は 50、apple 系は 30
TASK="packaging the items"
DATA=/home/robot/Dataset_and_Checkpoint/lerobot-dataset/<データセット名>
CKPT=/home/robot/Dataset_and_Checkpoint/checkpoints/<モデルディレクトリ>      # 直下に config.json、model.safetensors があるディレクトリ
CAMS="{cam_high: {type: ros_image, topic: /usb_cam_high/image_raw, width: 640, height: 480, fps: $FPS}, cam_low: {type: ros_image, topic: /usb_cam_low/image_raw, width: 640, height: 480, fps: $FPS}, cam_left_wrist: {type: ros_image, topic: /usb_cam_left_wrist/image_raw, width: 640, height: 480, fps: $FPS}, cam_right_wrist: {type: ros_image, topic: /usb_cam_right_wrist/image_raw, width: 640, height: 480, fps: $FPS}}"

echo $FPS $DATA $CKPT; echo "$CAMS"    # どれも空でないことを確認
```

`CAMS` にはカメラを 4 台とも書いておけば大丈夫です。モデルは学習時に使ったカメラだけを使い、それ以外のカメラはデータセットに記録されるだけです。

---

## 3. ロボットの起動（ターミナル 1、作業中は閉じない）

```bash
source ~/interbotix_ws_noetic/devel/setup.bash
roslaunch aloha 4arms_tactile_teleop.launch
```

4 本のアーム（`master_left`、`master_right`、`puppet_left`、`puppet_right`）と、4 台のカメラのトピック `/usb_cam_{high,low,left_wrist,right_wrist}/image_raw` が起動します。カメラは launch が占有しているので、LeRobot は `ros_image` タイプでトピックを読みます。画像はすでに RGB なので、BGR/RGB の変換は不要です。

カメラの確認：`rostopic hz /usb_cam_high/image_raw`。

以降のコマンドはすべて**別のターミナル**で実行します（先に 2.3 の初期化をしてください）。

---

## 4. グリッパのキャリブレーション（グリッパ交換や組み直しの後に 1 回）

```bash
python examples/aloha_ros/calibrate_master_gripper.py --side left
python examples/aloha_ros/calibrate_master_gripper.py --side right
python examples/aloha_ros/calibrate_puppet_gripper.py --side left
python examples/aloha_ros/calibrate_puppet_gripper.py --side right
```

各スクリプトで、グリッパを完全に閉じてから完全に開き、Ctrl+C で終了します。表示された min/max を `examples/aloha_ros/aloha_scripts/constants.py` の `MASTER_/PUPPET_<SIDE>_GRIPPER_JOINT_CLOSE / _OPEN` に書き込みます。フォロワー用のスクリプトは、手で動かせるように先にグリッパ（グリッパのみ）を PWM モードに切り替えてトルクを切ります。

注意：キャリブレーション値はグリッパの正規化を決めます。値を変えると、古い値で収録したデータや学習したモデルとはグリッパの意味がずれます。

---

## 5. 遠隔操作のテスト

```bash
python examples/aloha_ros/aloha_ros_teleop.py --fps $FPS                    # 双腕
python examples/aloha_ros/aloha_ros_teleop_one_arm.py --side right --fps $FPS   # 片腕
```

キー操作：→ 開始／一時停止からの再開、← 一時停止してロック（フォロワーは追従を止め、リーダーはその場で固定）、ESC または q で終了（4 本のアームがレスト姿勢に戻って電源オフ）。

LeRobot 標準のコマンドも使えます：

```bash
lerobot-teleoperate \
  --robot.type=aloha_ros --robot.init_ros_node=false --robot.fps=$FPS --robot.cameras="$CAMS" \
  --teleop.type=aloha_teleop_ros --teleop.fps=$FPS \
  --fps=$FPS --display_data=true
```

---

## 6. データ収集

### 6.1 ルート A：LeRobot データセットとして直接収録

**方法 1：本リポジトリのスクリプト。** 先に `examples/aloha_ros/aloha_ros_record.py` の冒頭にある設定（`NUM_EPISODES`、`EPISODE_TIME_SEC`、`TASK_DESCRIPTION`、`HF_REPO_ID`、`DATASET_ROOT`）を書き換えます。`DATASET_ROOT` にデータセットがすでにある場合は、その続きに追加で収録します（fps は同じである必要があります）。

```bash
python examples/aloha_ros/aloha_ros_record.py --fps $FPS
```

各エピソードの前に 4 本のアームが開始姿勢に戻ります。→ で収録開始。収録中は、→ で早めに終了して保存、← でこのエピソードを破棄して撮り直し、ESC で終了（収録中のエピソードは破棄）。

**方法 2：`lerobot-record`。**

```bash
lerobot-record \
  --robot.type=aloha_ros --robot.fps=$FPS --robot.cameras="$CAMS" \
  --teleop.type=aloha_teleop_ros --teleop.init_ros_node=false --teleop.fps=$FPS \
  --dataset.repo_id=local/<データセット名> --dataset.root=$DATA --dataset.no_stamp=true \
  --dataset.fps=$FPS --dataset.num_episodes=50 --dataset.episode_time_s=15 --dataset.reset_time_s=10 \
  --dataset.single_task="$TASK" --dataset.push_to_hub=false \
  --display_data=true
```

### 6.2 ルート B：ALOHA2 の生 hdf5

act-triple-plus-aloha2 リポジトリで収録します（本リポジトリの venv ではなく、そちらの conda 環境 `aloha` を使います）：

```bash
conda activate aloha
cd /mnt/server_ssd2/Programs_server/act-triple-plus-aloha2
python aloha_scripts/record_episodes_aloha2.py --task_name packaging_simple_2 --start_idx 0 --end_idx 49
```

タスクごとの保存先、エピソード長（`episode_len`）、カメラ一覧は、そのリポジトリの `aloha_scripts/constants.py` にある `TASK_CONFIGS` で設定します。詳しくはそのリポジトリの `aloha_scripts/ALOHA_SCRIPTS_USAGE_CN.md`（中国語）を参照してください。

---

## 7. hdf5 → LeRobot v3.0 の変換（ルート B）

データの収集と変換は act-triple-plus-aloha2 リポジトリにまとめてあります。変換スクリプトは `aloha_scripts/convert_aloha_hdf5_to_lerobot_v30.py` です。lerobot を import するので、act-triple の conda 環境 `aloha` ではなく、**本リポジトリの venv** で実行します：

```bash
source /mnt/server_ssd2/Programs_server/lerobot/.venv/bin/activate
cd /mnt/server_ssd2/Programs_server/act-triple-plus-aloha2
python aloha_scripts/convert_aloha_hdf5_to_lerobot_v30.py \
  --raw-dir /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2 \
  --output-dir /home/robot/Dataset_and_Checkpoint/lerobot-dataset/packaging_simple_2_v30 \
  --task "packaging the items" \
  --fps 50
```

- `--raw-dir` には**複数のディレクトリ**を指定でき、指定した順に 1 つのデータセットにまとめます。たとえば元のデモ + 追加収録 + `hil_rollout.py` で記録した HIL（先に `hil_split_episodes.py` で修正区間を切り出す）。同じディレクトリを 2 回書くと 2 回分変換されるので、少量のデータの比率を上げるのに使えます。`--task` は 1 つなら全ディレクトリ共通、ディレクトリごとに 1 つずつ指定することもできます。
- すべてのディレクトリでカメラ・画像サイズ・関節の次元が一致している必要があり、違うとエラーになります。
- 1 回の実行で v3.0 データセットができ、`meta/stats.json` には正確な分位点（q01/q10/q50/q90/q99）が計算済みです。LeRobot 標準の結合や統計の再計算は、エピソードごとの分位点の包絡しか残さず範囲が広めになるので、複数のデータは別々に変換して後で結合するのではなく、まとめて変換してください。
- `meta/source_episodes.tsv` に、各エピソードがどのディレクトリのどのファイルから来たかが記録されます。
- 関節名は `left_waist.pos … right_gripper.pos`（左腕が先）で、ロボット側の名前と一致しています。アクションは収録時の絶対値のまま保存されます。
- 色：画像は `cv2.imdecode` の結果をそのまま保存しています。openpi の `convert_aloha_data_to_lerobot_no_bgr2rgb.py` と照合して正しい RGB であることを確認済みなので、BGR→RGB 変換を**追加しないでください**。
- その他の引数：`--cameras` で一部のカメラだけ残す、`--episodes` で一部のエピソードだけ変換（ディレクトリが 1 つのときのみ）、`--use-images` で可逆 PNG として保存（約 40 倍の容量）、`--overwrite` で既存の出力を上書き。

データの確認：

```bash
lerobot-dataset-viz --repo-id local/packaging_simple_2_v30 \
  --root /home/robot/Dataset_and_Checkpoint/lerobot-dataset/packaging_simple_2_v30 --episode-index 0
```

---

## 8. 学習（サーバーまたはクラウド GPU で）

データセットを学習用マシンにコピーします：

```bash
rsync -a --info=progress2 /home/robot/Dataset_and_Checkpoint/lerobot-dataset/packaging_simple_2_v30 server:/path/to/datasets/
```

以下のコマンドは学習用マシンの lerobot 環境で実行し、`--dataset.root` はそのマシン上のパスに置き換えてください。

### 8.1 ACT（ルート A の例）

```bash
lerobot-train \
  --dataset.repo_id=local/aloha2_apple --dataset.root=$DATA \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=outputs/act_aloha2_apple --job_name=act_aloha2_apple \
  --steps=100000 --batch_size=8 --save_freq=10000 --wandb.enable=false
```

### 8.2 pi0.5（ルート B）

```bash
lerobot-train \
  --dataset.repo_id=local/packaging_simple_2_v30 --dataset.root=/path/to/packaging_simple_2_v30 \
  --dataset.image_transforms.enable=true \
  --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
  --policy.dtype=bfloat16 --policy.gradient_checkpointing=true \
  --policy.device=cuda --policy.push_to_hub=false \
  --batch_size=32 --steps=20000 --save_freq=5000 --num_workers=8 \
  --output_dir=outputs/pi05_packaging_simple_2_default --job_name=pi05_packaging_simple_2_default \
  --wandb.enable=false
```

batch 32 のフルファインチューニングには大容量 VRAM の GPU が必要で、3090 のような 24 GB の GPU では足りません。

### 8.3 MolmoAct2（ルート B）

```bash
lerobot-train \
  --dataset.repo_id=local/packaging_simple_2_v30 --dataset.root=/path/to/packaging_simple_2_v30 \
  --dataset.image_transforms.enable=true \
  --policy.type=molmoact2 --policy.checkpoint_path=allenai/MolmoAct2 \
  --policy.image_keys='["observation.images.cam_high","observation.images.cam_left_wrist","observation.images.cam_right_wrist"]' \
  --policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.compile_model=true \
  --policy.push_to_hub=false \
  --batch_size=8 --steps=20000 --save_freq=2000 \
  --output_dir=outputs/molmoact2_packaging --job_name=molmoact2_packaging --wandb.enable=false
```

### 8.4 学習結果と学習の再開

- checkpoint は `<output_dir>/checkpoints/<ステップ数>/pretrained_model/` にあります（`last` は最新のものを指します）。デプロイに必要なのは `pretrained_model` ディレクトリだけなので、それをロボットにコピーします：
  ```bash
  rsync -a server:/path/outputs/pi05_xxx/checkpoints/020000/pretrained_model/ \
    /home/robot/Dataset_and_Checkpoint/checkpoints/pi05_xxx_020000/
  ```

  `.tar.gz` にまとめて転送した場合は、展開する前に `gzip -t xxx.tar.gz` でファイルが完全か確認してください（ダウンロードが途中で切れたアーカイブでも一部のファイルは一覧に出ますが、展開すると `Unexpected EOF` になります）。
- 中断した学習の再開：
  ```bash
  lerobot-train --config_path=outputs/<job>/checkpoints/last/pretrained_model/train_config.json --resume=true
  ```

---

## 9. デプロイ・実行

入口は 2 つあります：

| | `aloha_ros_eval.py` | `lerobot-rollout` |
| --- | --- | --- |
| 推論方式 | 同期（chunk を実行し終えてから次を推論） | 同期、または RTC |
| キー操作 | → / ← / ESC、エピソード単位で実行 | strategy による（9.2 参照） |
| データ記録 | 任意（`--dataset_root`） | episodic、dagger は記録必須 |

### 9.1 `aloha_ros_eval.py`（同期、普段の評価におすすめ）

```bash
# ルート B のモデル（pi05、MolmoAct2）
python examples/aloha_ros/aloha_ros_eval.py --data_format aloha2 --fps 50 \
  --policy_path $CKPT --task "$TASK" --num_episodes 5 --episode_time_s 180

# ルート A のモデル（ACT など）
python examples/aloha_ros/aloha_ros_eval.py --fps 30 --policy_path $CKPT --num_episodes 5 --episode_time_s 30
```

キー操作（スクリプトを実行しているターミナルで押す）：

| キー | 動作 |
| --- | --- |
| →（または `n`） | 開始姿勢で待機中：このエピソードを開始。実行中：早めに終了（完了扱い） |
| ←（または `r`） | このエピソードを破棄し、開始姿勢に戻ってやり直し。`--num_episodes` には数えない |
| ESC（または `q`） | 終了：アームが 2 段階でレスト姿勢に戻って電源オフ |

- 実行中は 1 行のプログレスバー（経過秒数と直近の推論時間）だけが表示されます。
- 起動時に 1 回ウォームアップの推論を行います（pi05 で約 10〜20 秒）。その後の通常の推論は約 0.2 秒（3090）です。同期推論では chunk を実行し終えるたびに推論待ちで一瞬止まりますが、これは正常です。連続した動きにしたい場合は 9.3 の RTC を使います。
- スクリプトはモデルの入力と関節の順序が `--data_format` と合っているかを確認し、合わなければエラーにして、使うべき形式を表示します。
- 動画はデフォルトでモデルのディレクトリに保存されます（スクリプト冒頭の `SAVE_VIDEOS`、`VIDEO_OUTPUT_DIR`）。`--dataset_root <ディレクトリ>` を付けると、評価の様子を LeRobot データセットとしても保存します。
- ACT で時間的アンサンブルを使うには、スクリプト冒頭の `POLICY_TEMPORAL_ENSEMBLE_COEFF` を 0.01、`POLICY_N_ACTION_STEPS` を 1 にします。
- 起動時の `[ERROR] Please set the gripper's 'operating mode' to 'pwm' or 'current'` は Interbotix がグリッパ初期化時に出すメッセージなので、無視して構いません。

### 9.2 `lerobot-rollout` のモード

| `--strategy.type` | 操作方法 | データ記録 |
| --- | --- | --- |
| `base` | キー操作なし。`--duration` 秒経過するか Ctrl+C で終了 | しない |
| `base` + `--interactive=true` | ターミナルに `/start` で開始、`/reset` で停止して初期姿勢へ、`/stop` で終了 | しない |
| `episodic` | → でこのエピソードまたはリセット期間を早めに終了、← で破棄して撮り直し、ESC で終了 | 必須 |
| `dagger` | キーボードまたはフットペダルで人が介入（10 章参照） | 必須、リーダーアームが必要 |

- データを記録するモードでは、`--dataset.repo_id` の名前を `rollout_` で始める必要があります。
- 既存のデータセットに追加で記録するときは `--resume=true` を付けます。カメラと fps は元のデータセットと同じでなければなりません。
- ルート B のモデルには必ず `--robot.gripper_state_format=aloha2` を付けます。ルート A では付けません。
- ACT で時間的アンサンブルを使うには `--policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1`。

ポリシーを動かすだけで記録しない場合（同期推論）：

```bash
lerobot-rollout --strategy.type=base \
  --policy.path=$CKPT \
  --robot.type=aloha_ros --robot.fps=$FPS --robot.gripper_state_format=aloha2 --robot.cameras="$CAMS" \
  --fps=$FPS --duration=180 --task="$TASK" --display_data=true
```

### 9.3 RTC リアルタイムチャンキング（オプション）

**オプション**：RTC を使わなくても普通に動きます（9.1、9.2 はどちらも同期推論）。pi05 のように推論が遅いモデルで、chunk 間の停止が気になる場合にだけ使います。

論文：Kevin Black, Manuel Y. Galliker, Sergey Levine. *Real-Time Execution of Action Chunking Flow Policies*, 2025. <https://arxiv.org/abs/2506.07339>（Physical Intelligence の紹介ページ：<https://www.physicalintelligence.company/research/real_time_chunking>）

RTC は次の chunk をバックグラウンドで先に計算し、新しい chunk の先頭を前の軌道に滑らかにつなげるので、実行中に止まりません。`lerobot-rollout` でのみ使えます。以下はキー操作ありでデータも記録する episodic 版です：

```bash
lerobot-rollout --strategy.type=episodic \
  --policy.path=$CKPT \
  --inference.type=rtc --inference.rtc.execution_horizon=30 --inference.queue_threshold=30 \
  --inference.rtc.max_guidance_weight=10.0 \
  --robot.type=aloha_ros --robot.fps=50 --robot.gripper_state_format=aloha2 --robot.cameras="$CAMS" \
  --dataset.repo_id=local/rollout_pi05_eval \
  --dataset.root=/home/robot/Dataset_and_Checkpoint/lerobot-dataset/rollout_pi05_eval \
  --dataset.num_episodes=5 --dataset.episode_time_s=180 --dataset.reset_time_s=30 \
  --dataset.push_to_hub=false --dataset.single_task="$TASK" \
  --fps=50 --task="$TASK"
```

記録しない場合：`--strategy.type=base` にして `--dataset.*` をすべて外し、`--duration=180` を付けます。

パラメータの決め方（50 Hz、pi05、3090 での実測）：

- guided RTC はデノイズの各ステップで逆伝播が 1 回増えるため、推論は約 240〜290 ms（通常の推論は 200 ms）。画像取得と前処理を含めると、遅延は約 15 tick です。
- `execution_horizon` はこの遅延より十分大きくする必要があります。デフォルトの 10 だと、新旧 chunk の切り替わり点がガイダンスの範囲外になり、切り替わるたびにガクッと跳ねて、かえって同期推論より悪くなります。
- `queue_threshold=30`：キューの残りが 30 ステップになったら次の chunk の計算を始めます。計算が終わる頃には約 15 ステップ実行済みなので、切り替わり点は 15〜30 ステップの遷移区間に入ります。GPU で他のプログラムも動いていて遅延が大きくなる場合は、両方を 35 に上げてください。
- 学習時に RTC を有効にしていない checkpoint（`rtc_training_max_delay=0`、現在のモデルはすべてこれ）は、デフォルトの guided モードしか使えません。

---

## 10. HIL：人の介入による追加データ収集（DAgger）

ポリシーが自律で動き、失敗しそうになったら人がリーダーアームで介入して修正し、その修正区間を学習データに加えて学習し直します。方法は 2 通りあり、**どちらも使え、混ぜて使うこともできます**：

| | 方法 A：`lerobot-rollout` の dagger | 方法 B：act-triple の `hil_rollout.py` |
| --- | --- | --- |
| 改善したいモデル | LeRobot で学習した checkpoint（pi05、MolmoAct2、ACT）、このマシンの GPU で推論 | openpi または N0-VTLA のモデル、サーバーの `serve_policy.py` で推論 |
| 実行環境 | 本リポジトリの venv | act-triple の conda 環境 `aloha` |
| 記録する範囲 | 修正区間だけ、修正 1 回 = 1 エピソード | エピソード全体を記録し、各フレームに phase / intervention ラベルを付け、後で切り出す |
| データ形式 | LeRobot v3.0 データセット（`rollout_*`） | ALOHA2 の生 hdf5（N0 は VTLA ディレクトリ） |
| 学習に使える先 | LeRobot のみ | LeRobot、ACT（act-triple）、openpi のどれでも |
| LeRobot 学習用データセットへの追加 | `merge_hil_into_lerobot_v30.py`（10.1） | `hil_split_episodes.py` で修正区間を切り出し、`convert_aloha_hdf5_to_lerobot_v30.py` でデモと一緒に変換（10.2） |

選び方：改善したいのが LeRobot で学習したモデルなら方法 A、openpi / N0 のモデルなら方法 B です。DAgger の修正はそのとき動いていたモデルの失敗に対するものなので、同じモデルの学習に使うのが最も効果的です。別のモデルの学習に使っても有効なデモデータにはなりますが、そのモデル自身の失敗を修正したものではありません。方法 B のデータは生の形式なので、後でどのフレームワークにも使えます。方法 A のデータは LeRobot でしか使えず、hdf5 に戻すツールはありません。

どちらの方法も介入はクラッチ式で、キー操作も同じです：→ 一時停止／ポリシーに戻す、← 修正開始／修正終了。

### 10.1 方法 A：`lerobot-rollout` の dagger（LeRobot のモデル）

修正区間だけを記録し、修正 1 回が 1 エピソードとして `intervention` フラグ付きで保存されます（修正モードではすべて True）。

介入方式には 2 つのバージョンがあり、違いは「修正開始」の瞬間にリーダーとフォロワーをどう引き継ぐかだけです：

| | バージョン 1：変更前（非クラッチ、フォロワーをブレンド） | バージョン 2：変更後（クラッチ式） |
| --- | --- | --- |
| リーダーの解放 | 0.6 秒かけて柔らかくしてからトルクオフ | 0.3 秒かけて柔らかくしてからトルクオフし、リーダーが静止するのを待つ（最大 1 秒） |
| フォロワーの追従 | 0.3 秒かけて保持していた姿勢からリーダーの**絶対**位置へブレンド | リーダーとフォロワーの差分を記録し、「リーダー − 差分」に追従 |
| 手を添えたリーダーが沈んだとき | フォロワーもリーダーの位置まで沈み、少し動く | 差分が沈み込みを打ち消すので、フォロワーは動かない |
| 記録しない先頭区間 | 0.5 秒 | 0.2 秒 |
| 主なパラメータ | `correction_blend_s=0.3`、`correction_skip_s=0.5`、`release_ramp_s=0.6` | `relative_on_release=true`、`correction_blend_s=0`、`correction_skip_s=0.2`、`release_ramp_s=0.3` |

どちらのバージョンでも記録されるデータの形式は同じなので、同じデータセットに記録できます。2 つの介入方式の効果を比べたい場合は、`--dataset.root` を分けてください。既存の HIL データセットに追加で記録するときは `--resume=true` を付けます。

#### バージョン 1：変更前（非クラッチ、フォロワーをブレンド）

```bash
lerobot-rollout --strategy.type=dagger --strategy.record_autonomous=false \
  --strategy.keyboard.pause_resume=right --strategy.keyboard.correction=left \
  --strategy.release_teleop_on_resume=false --strategy.num_episodes=20 \
  --strategy.correction_blend_s=0.3 --strategy.correction_skip_s=0.5 \
  --robot.type=aloha_ros --robot.fps=$FPS --robot.gripper_state_format=aloha2 --robot.cameras="$CAMS" \
  --teleop.type=aloha_teleop_ros --teleop.init_ros_node=false --teleop.fps=$FPS \
  --teleop.release_on_connect=false --teleop.release_ramp_s=0.6 \
  --policy.path=$CKPT \
  --fps=$FPS --dataset.repo_id=local/rollout_packaging_hil --dataset.no_stamp=true \
  --dataset.root=/home/robot/Dataset_and_Checkpoint/lerobot-dataset/rollout_packaging_hil \
  --dataset.fps=$FPS --dataset.single_task="$TASK" --dataset.push_to_hub=false \
  --display_data=true
```

#### バージョン 2：変更後（クラッチ式）

```bash
lerobot-rollout --strategy.type=dagger --strategy.record_autonomous=false \
  --strategy.keyboard.pause_resume=right --strategy.keyboard.correction=left \
  --strategy.release_teleop_on_resume=false --strategy.num_episodes=20 \
  --strategy.correction_blend_s=0 --strategy.correction_skip_s=0.2 \
  --robot.type=aloha_ros --robot.fps=$FPS --robot.gripper_state_format=aloha2 --robot.cameras="$CAMS" \
  --teleop.type=aloha_teleop_ros --teleop.init_ros_node=false --teleop.fps=$FPS \
  --teleop.release_on_connect=false --teleop.relative_on_release=true --teleop.release_ramp_s=0.3 \
  --policy.path=$CKPT \
  --fps=$FPS --dataset.repo_id=local/rollout_packaging_hil --dataset.no_stamp=true \
  --dataset.root=/home/robot/Dataset_and_Checkpoint/lerobot-dataset/rollout_packaging_hil \
  --dataset.fps=$FPS --dataset.single_task="$TASK" --dataset.push_to_hub=false \
  --display_data=true
```

#### 操作手順とパラメータ

キー操作は両バージョン共通です：

1. 起動するとポリシーが自律で動き、リーダーアームはロックされたまま動きません。
2. 失敗しそうになったら **→ で一時停止**：フォロワーが止まり、リーダーがフォロワーの現在の姿勢まで動かされます。
3. リーダーを手で握り、**← で修正開始**：リーダーが徐々に柔らかくなってから解放され、上の表の方式で引き継いだ後、記録が始まります。
4. 修正が終わったら **← で修正終了**：この区間が 1 エピソードとして保存されます。
5. **→ でポリシーに戻す**と、自律動作を再開します。ESC でセッション全体を終了します。

| パラメータ | 役割 |
| --- | --- |
| `--teleop.relative_on_release` | `true` = クラッチ式（バージョン 2）：リーダーを解放して静止するのを待ってから、リーダーとフォロワーの差分を記録し、フォロワーは「リーダー − 差分」に追従。付けない（デフォルト `false`）= バージョン 1 で、フォロワーはリーダーの絶対位置にそのまま追従 |
| `--teleop.release_ramp_s` | この時間をかけてリーダーの位置ゲインを 0 まで下げてからトルクを切り、急に脱力しないようにする |
| `--strategy.correction_blend_s` | 修正開始時にフォロワーをリーダーの位置へブレンドする時間。バージョン 1 ではこれでフォロワーの急な跳びを防ぐ。クラッチ式ではフォロワーはもともと動かないので 0 |
| `--strategy.correction_skip_s` | 各修正の先頭のこの秒数は記録しない（引き継ぎ時の過渡的な動きを避けるため）。これより短い修正は保存もカウントもされない |
| `--strategy.release_teleop_on_resume=false` | ポリシーに戻した後もリーダーをロックしたままにし、トルクオフで落下しないようにする |
| `--strategy.num_episodes` | この回数の修正を記録したら自動で終了 |

#### 記録の後：データ処理と学習

dagger は自分のデータセット（上の `rollout_packaging_hil`）にだけ書き込みます。修正が 1 回終わるたびに 1 エピソードとして保存し、そのデータセット自身の `meta/` も更新します。**学習に使っているデータセットには手を付けません。** このデータを使うには、まず新しいデータセットに結合してから学習します。

**ステップ 1：結合。** act-triple の `aloha_scripts/merge_hil_into_lerobot_v30.py` を使います（本リポジトリの venv で実行）：

```bash
source /mnt/server_ssd2/Programs_server/lerobot/.venv/bin/activate
cd /mnt/server_ssd2/Programs_server/act-triple-plus-aloha2
python aloha_scripts/merge_hil_into_lerobot_v30.py \
  --base /home/robot/Dataset_and_Checkpoint/lerobot-dataset/packaging_simple_2_v30 \
  --hil /home/robot/Dataset_and_Checkpoint/lerobot-dataset/rollout_packaging_hil \
  --output-dir /home/robot/Dataset_and_Checkpoint/lerobot-dataset/packaging_simple_2_v30_hil
```

- 出力は新しいデータセットで、`--base` と `--hil` はどちらも変更されません。`--hil` は複数指定でき、同じものを 2 回書くと 2 回分追加されるので、HIL の比率を上げられます。終了時に、全フレームに占める HIL フレームの割合が表示されます。
- 2 種類のデータセットは特徴量が違うので、スクリプトが揃えてから結合します：
  - HIL 側の `intervention` と `cam_low` を削除。どのエピソードが HIL 由来かは `meta/source_episodes.tsv` に記録します。
  - base 側の `observation.velocity`、`observation.effort` を削除。dagger はこの 2 つを記録せず、pi05・MolmoAct2・ACT も `observation.state` しか使いません。
  - `robot_type` は base のものに統一。
- 動画は再エンコードせずにそのまま連結し、`meta/stats.json` は結合後の全データから正確な分位点を計算し直します。
- fps、関節名の順序、画像サイズは一致している必要があり、違うとエラーになります。HIL は `--robot.gripper_state_format=aloha2` を付けて記録したものでなければなりません（上の 2 つのコマンドには付いています）。この点はスクリプトでは確認できません。

**ステップ 2：学習。** 2 通りあります：

- 最初から学習：8 章のコマンドのデータセットを結合後のものに置き換えます。
- 既存のモデルから続けて学習：
  ```bash
  lerobot-train --policy.path=$CKPT \
    --dataset.repo_id=local/packaging_simple_2_v30_hil --dataset.root=/path/to/packaging_simple_2_v30_hil \
    --dataset.image_transforms.enable=true \
    --batch_size=32 --steps=5000 --save_freq=1000 \
    --output_dir=outputs/pi05_packaging_hil --job_name=pi05_packaging_hil --wandb.enable=false
  ```
  `--policy.path` で既存モデルの重みから始め、正規化の統計は新しいデータセットのものに自動で置き換わります。ここで `--resume=true` は使わないでください。これは元の学習を元のデータセットのまま再開するためのものです。既存モデルの入力に velocity/effort があり新しいデータセットにはなくても学習でき、この流れは ACT で実際に確認済みです。

### 10.2 方法 B：act-triple の `hil_rollout.py`（openpi / N0 のモデル）

**収録**（先にサーバーで `serve_policy.py` を起動しておく。キー操作・プログレスバー・パラメータは act-triple の `aloha_scripts/README_run_robot_ja.md` 第 1・3 節を参照）：

```bash
conda activate aloha
cd /mnt/server_ssd2/Programs_server/act-triple-plus-aloha2
python3 aloha_scripts/hil_rollout.py --task_name packaging_simple_2 --policy openpi \
  --host 10.240.77.20 --port 8000 --prompt "packaging the items" --num_episodes 20
```

データは `/home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2_hil/episode_N.hdf5` に保存されます。形式はデモの hdf5 と同じで、各フレームの HIL ラベルが追加されています。

**ステップ 1：修正区間を切り出す**（`aloha` 環境のまま）。エピソード全体のうち約 3 割は一時停止なので、そのまま学習には使えません：

```bash
python3 aloha_scripts/hil_split_episodes.py \
  /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2_hil \
  /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2_hil_corrections --keep intervention
```

**ステップ 2（任意）：比率を測る。** 修正区間が元のデモに対して何割か、10% / 20% にするには何回コピーすればよいかを表示します：

```bash
python3 aloha_scripts/hil_compare_lengths.py \
  /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2 \
  /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2_hil
```

**ステップ 3：デモと一緒に LeRobot v3.0 に変換**（本リポジトリの venv に切り替える）。修正区間の比率が低すぎる場合は、`_hil_corrections` を何回か書きます（1 回書くごとに 1 回分増えます）：

```bash
source /mnt/server_ssd2/Programs_server/lerobot/.venv/bin/activate
cd /mnt/server_ssd2/Programs_server/act-triple-plus-aloha2
python aloha_scripts/convert_aloha_hdf5_to_lerobot_v30.py \
  --raw-dir /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2 \
            /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2_hil_corrections \
  --output-dir /home/robot/Dataset_and_Checkpoint/lerobot-dataset/packaging_simple_2_hil_v30 \
  --task "packaging the items" --fps 50
```

すべて hdf5 由来なので、このデータセットは velocity/effort を残しており、7 章で変換したデータセットと特徴量がまったく同じです。

**ステップ 4：学習。** 10.1 のステップ 2 と同じく、最初から学習するか `--policy.path` で既存モデルから続けます。データセットは `packaging_simple_2_hil_v30` に置き換えます。

同じ切り出し済みの修正区間は、act-triple の ACT（hdf5 を直接読む）や openpi（openpi の変換スクリプトで v2.1 に変換）の学習にもそのまま使えます。手順は act-triple の `aloha_scripts/README_run_robot_ja.md` 第 4 節を参照してください。

### 10.3 両方を一緒に使う

できます。まず変換スクリプトでデモと方法 B の修正区間を一緒に v3.0 に変換し、その上に結合スクリプトで方法 A のデータセットを加えます：

```bash
source /mnt/server_ssd2/Programs_server/lerobot/.venv/bin/activate
cd /mnt/server_ssd2/Programs_server/act-triple-plus-aloha2
python aloha_scripts/convert_aloha_hdf5_to_lerobot_v30.py \
  --raw-dir /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2 \
            /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2_hil_corrections \
  --output-dir /home/robot/Dataset_and_Checkpoint/lerobot-dataset/packaging_simple_2_hil_v30 \
  --task "packaging the items" --fps 50
python aloha_scripts/merge_hil_into_lerobot_v30.py \
  --base /home/robot/Dataset_and_Checkpoint/lerobot-dataset/packaging_simple_2_hil_v30 \
  --hil /home/robot/Dataset_and_Checkpoint/lerobot-dataset/rollout_packaging_hil \
  --output-dir /home/robot/Dataset_and_Checkpoint/lerobot-dataset/packaging_simple_2_hil_all_v30
```

- 順番は逆にできません：hdf5 は変換スクリプト、LeRobot データセットは結合スクリプトしか通せません。
- 2 種類のデータを一緒にできるのは、関節の順序（左腕が先）、グリッパ状態の意味（ALOHA2 の qpos）、50 Hz がすべて一致し、2 つのリポジトリのグリッパのキャリブレーション定数もまったく同じだからです。
- 方法 A のデータを加えた時点で、結果には velocity/effort がなくなります。
- 逆方向はできません：方法 A のデータは act-triple の ACT や openpi には使えません。

---

## 11. スリープ／電源オフ

```bash
python examples/aloha_ros/aloha_scripts/sleep_modify.py --sleep             # 4 本のアームをレスト姿勢へ、通電したまま（休憩時）
python examples/aloha_ros/aloha_scripts/sleep_modify.py --shut_down         # 4 本のアームをレスト姿勢へ戻してから電源オフ（作業終了時）
python examples/aloha_ros/aloha_scripts/sleep_modify.py --shut_down_puppet  # フォロワーだけレスト姿勢へ戻して電源オフ
```

レスト姿勢へは 2 段階で移動し（まず旧レスト姿勢、次に実測した最終姿勢）、到着してからトルクを切ります。アームが落下してぶつかるのを防ぐためです。

---

## 12. よくある問題

| 症状 | 原因と対処 |
| --- | --- |
| `Couldn't parse '' into an int` | `$FPS`、`$CAMS` などの変数が今のターミナルで空になっている。2.4 に従って設定し直す |
| `No module named 'rospkg'` | `uv sync` で ROS の依存パッケージが削除された。`uv pip install rospkg catkin_pkg netifaces h5py` を再実行 |
| グリッパの動きがおかしい（開くべきときに開かない、閉じるべきときに閉じない） | データのルートとデプロイ時のパラメータが合っていない（0 章参照）。またはキャリブレーションが変わった |
| 同期推論で毎秒一瞬止まる | 正常。chunk を実行し終えるたびに推論を待つため。連続した動きにしたければ RTC を選択 |
| RTC で周期的に跳ねる | `execution_horizon` が小さすぎる。9.3 参照 |
| `--resume=true` での追加記録に失敗する | カメラまたは fps が既存のデータセットと違う。`CAMS` にはデータセットにあるカメラをすべて書く |
| `.tar.gz` の展開で `Unexpected EOF` | ダウンロードが不完全。ダウンロードし直し、先に `gzip -t` で確認する |
| 起動時の `Please set the gripper's 'operating mode'` | Interbotix のメッセージなので無視してよい |
