# Real-Time Secure Face Recognition Attendance System Using Raspberry Pi

## 1. Project Overview

This project implements a real-time faculty attendance system. A **Raspberry Pi 3B+** with a Logitech HD 720p USB webcam runs as an attendance **kiosk**: it recognises faculty on the device using lightweight ONNX models, with no external cloud computing. A separate **admin PC** runs the management dashboard, registers new faculty, trains the face data and holds the master database. The two machines synchronise automatically over the local network.

Key properties:

* Passive anti-spoofing (MiniFAS) tells real faces from photographs and screens without asking the person to turn their head.
* Registration is a guided, no-terminal flow on the admin PC. Any failure rolls the system back to how it was.
* The kiosk keeps working if the admin PC is off or the network is down. Scans are queued and delivered later with their real time.
* A face is accepted only when it clearly matches one person; look-alikes are rejected rather than marked as the wrong person.

## 2. Architecture

```text
              ADMIN PC (master)                              RASPBERRY PI (kiosk)
  ┌───────────────────────────────────────┐          ┌──────────────────────────────────┐
  │ Flask dashboard + admin login         │          │ Webcam -> YuNet -> MiniFAS ->    │
  │ Registration (camera on the admin PC) │          │ SFace -> match -> attendance     │
  │ Training                              │          │                                  │
  │ Master database  (attendance.db)      │  HTTP    │ Local copy of faculty list and   │
  │ known_faces/  +  classifier (.pkl)    │◄────────►│ classifier (downloaded)          │
  │ Kiosk sync API (kiosk_api.py)         │  (LAN)   │ Local database + upload queue    │
  └───────────────────────────────────────┘          └──────────────────────────────────┘
        faculty list + classifier  ───────────────────────►  new faces arrive in seconds
        ◄───────────────────────  attendance scans (queued if offline, delivered later)
```

| Data | Lives on | Direction |
| --- | --- | --- |
| Faculty records, face photos, trained classifier | Admin PC (master) | Admin PC → kiosk |
| Attendance records and reports | Admin PC (master) | Kiosk → admin PC |
| Recent attendance (last 8 days) and the upload queue | Kiosk | Local only |

### How the link works

* The kiosk calls the admin PC every few seconds (default 5, and again after about 1 second whenever something changed) and sends its queued scans together with its status.
* Each kiosk has its own secret key, created on the dashboard's **Kiosks** page. Every request and every reply is signed with HMAC-SHA256. The trained classifier is a pickle file, so the kiosk never loads it unless the signature and checksum verify.
* Each scan has a unique id, so a scan delivered twice (for example after a lost reply) is counted once.
* The faculty list and classifier are downloaded and installed only after verification. A failed download leaves the previous working data in place.
* Clock check: a kiosk clock more than 2 minutes off shows a warning on the kiosk; more than 5 minutes blocks syncing until the clock is fixed.

## 3. Hardware and Software Stack

### Hardware

* Kiosk: Raspberry Pi 3B+, Logitech HD 720p USB webcam, microSD card (16 GB minimum), stable 2.5 A power supply.
* Admin PC: any Windows / Linux / macOS computer with a webcam, on the same network as the kiosk.

### Software and models

* Python 3, OpenCV (DNN module, 4.8 or newer), NumPy, SQLite
* Flask (admin PC only)
* YuNet (face detection), SFace (128-dimensional face embeddings), MiniFAS (passive anti-spoofing)
* scikit-learn: only needed to read a classifier created by an older version of this project (see section 8)

## 4. Features

* Real-time face detection and recognition on the Pi.
* Passive liveness detection against printed photos and phone screens.
* Attendance state machine: first scan of the day is a check-in (late after `COLLEGE_START_TIME`, default 10:00:00), the next scan is a check-out, a later scan is a re-entry, and repeats inside `COOLDOWN_SEC` (default 60) are ignored.
* Guided faculty registration with automatic rollback on any error.
* Admin login with password hashing and brute-force lockout.
* Web dashboard: faculty management, today's attendance, monthly reports with CSV export, and a Kiosks page showing which kiosks are online and up to date.
* Offline-tolerant kiosk with an automatic catch-up queue.
* Settings tuned for limited hardware (1 GB RAM).

## 5. How a Face Is Recognised

SFace turns each face into a 128-dimensional vector $\mathbf{e}$ of length 1. The similarity of two faces is the cosine of the angle between their vectors:

$$
\cos(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert} = \mathbf{a}\cdot\mathbf{b}
\qquad\text{(unit vectors, so the denominator is 1)}
$$

Each enrolled person $p$ has several stored reference vectors $\mathbf{r}_{p,j}$ (the original photos plus lighting and rotation variants). For a live scan $\mathbf{e}$ (the average of `EMB_FRAMES` frames):

$$
s_p = \max_j \; \mathbf{e}\cdot\mathbf{r}_{p,j}
$$

Let $p^*$ be the person with the highest score $s_{p^*}$ and $q$ the runner-up. The scan is accepted only if both conditions hold:

$$
s_{p^*} \ge \texttt{COSINE\_THRESHOLD}\;(0.363)
\qquad\text{and}\qquad
s_{p^*} - s_{q} \ge \texttt{MATCH\_MARGIN}\;(0.05)
$$

* The first condition rejects strangers. 0.363 is the same-person threshold published for SFace.
* The second condition rejects ambiguous matches: when two people are almost equally close, nobody is marked.
* Neither condition depends on how many people are enrolled, so accuracy does not degrade as faculty are added. (Earlier versions gated on an SVM probability, which shrinks as the number of classes grows and caused later registrants to be rejected.)

Full per-frame pipeline on the kiosk:

```text
frame -> YuNet (exactly one face, at least MIN_FACE_PX wide)
      -> MiniFAS passive anti-spoof (median over TEMPORAL_FRAMES frames must be REAL)
      -> SFace embedding, EMB_FRAMES frames; frames must show the same face
      -> cosine match + margin check
      -> attendance state machine -> local database + upload queue (one transaction)
```

## 6. Project Structure

```text
frs_attendance/
├── admin/                      # runs on the admin PC
│   ├── app.py                  # Flask dashboard, login, registration pages, Kiosks page
│   ├── registration.py         # guided registration job manager (capture, train, rollback)
│   ├── enroll.py               # native camera window that captures the 3 photos
│   ├── enroll_lock.py          # lock file + heartbeat used during capture
│   ├── train_lite.py           # builds the face-data file from known_faces/
│   ├── kiosk_api.py            # sync API used by kiosks (signed requests)
│   ├── create_admin.py         # create / reset / list admin accounts
│   ├── templates/              # HTML templates
│   ├── known_faces/            # reference photos (created at runtime, do not commit)
│   ├── attendance.db           # master database (created at runtime, do not commit)
│   └── face_classifier_lite.pkl  # trained face data (created at runtime)
├── common/                     # shared by both machines (must be identical on both)
│   ├── lite_core.py            # database, camera, face engine, matching, attendance logic
│   ├── kiosk_proto.py          # request/reply signing and fingerprints
│   ├── anti_spoof.py           # MiniFAS passive anti-spoofing
│   └── models/                 # ONNX models: YuNet, SFace, MiniFAS
├── kiosk/                      # runs on the Raspberry Pi
│   ├── kiosk.py                # recognition and attendance loop
│   ├── kiosk_sync.py           # background link to the admin PC, offline queue
│   ├── preflight.py            # system and model verification
│   ├── kiosk_config.json       # created from the Kiosks page (do not commit)
│   ├── attendance.db           # local copy + upload queue (created at runtime)
│   └── face_classifier_lite.pkl  # downloaded from the admin PC
└── README.md
```

Code is shared from `common/`; each app keeps its own data in the folder of the script you start (`admin/app.py` uses `admin/`, `kiosk/kiosk.py` uses `kiosk/`). This also lets you run both on one computer without them sharing a database. Override with `FRS_DATA_DIR` and `FRS_MODELS_DIR` if needed.

## 7. Installation and Setup

### 7.1 Clone the repository

On **both** machines:

```bash
git clone <repository_url>
cd frs_attendance
```

### 7.2 Python environments

Admin PC:

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install "Flask>=3.0" "opencv-python>=4.8" numpy
```

Raspberry Pi:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "opencv-python>=4.8" numpy
```

### 7.3 Verify the ONNX models

If the project was cloned without Git LFS, model files can be tiny text pointers instead of the real binaries. Check the size of every file in `common/models/`:

```bash
ls -lh common/models/
```

SFace should be about 36.8 MB. If a file is only some bytes, download it directly:

```bash
wget -O common/models/face_recognition_sface_2021dec.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

The YuNet and SFace files must be **byte-identical** on both machines. The kiosk compares fingerprints and refuses face data built with different models.

### 7.4 Create the first admin account (admin PC)

```bash
python admin/create_admin.py
```

Other options: `--reset USERNAME`, `--list`. Passwords need at least 8 characters.

### 7.5 Start the dashboard (admin PC)

Start it from a normal desktop session, because registration opens a camera window.

```bash
python admin/app.py
```

The console prints the dashboard address and the address kiosks should use (the admin PC's LAN address, port 5000). Allow that port through the admin PC's firewall.

### 7.6 Link the kiosk

1. Log in to the dashboard and open **Kiosks**.
2. Enter a name (for example `Main gate`) and click **Create key**.
3. Copy the text shown (it is displayed only once; use **New key** if it is lost) into `kiosk/kiosk_config.json` on the Pi:

```json
{
  "admin_url": "http://192.168.1.20:5000",
  "kiosk_id": "k-775af580",
  "secret": "<long random key shown on the Kiosks page>",
  "poll_seconds": 5
}
```

1. Copy `common/models/` to the Pi if it is not already there, then run the checks and start the kiosk:

```bash
python kiosk/preflight.py
python kiosk/kiosk.py
```

The top bar of the kiosk shows `Synced`, and the Kiosks page shows `Up to date`. Without a `kiosk_config.json` the kiosk runs stand-alone on its own local data.

### 7.7 Moving from a single-machine install

Copy the old `attendance.db`, `known_faces/` and `face_classifier_lite.pkl` into `admin/` **before** linking the kiosk. A kiosk that still holds faculty data will not overwrite itself from an empty admin PC; it shows a warning instead.

## 8. Usage Workflow

### Registering a faculty member (admin PC)

1. Open **Faculty Management** and choose Register.
2. Enter the details and click **Start Native Face Enrollment**. A camera window opens on the admin PC.
3. Press **SPACE** for each of the 3 photos (straight, slightly left, slightly right). **ESC** cancels.
4. Review the photos and click **Train & save face data**.
5. When the page shows "Face has been captured, trained and saved.", click **Register Faculty** to finish.

Checks and safeguards:

* All 3 photos must contain exactly one face and show the same person.
* The face must not already be enrolled under another ID.
* After training, the new person is test-matched against everyone else exactly as the kiosk will do it. If the face is too similar to someone already enrolled, or fewer than 2 photos were usable, the registration is refused.
* Any error, timeout or restart rolls back the database, photos and face data. **Cancel & undo** reverses a trained registration until it is finished.
* The new face is available to kiosks as soon as training completes (within seconds). It is final after **Register Faculty**.

### Daily operation

* Keep `admin/app.py` running to receive attendance and to register people. The kiosk does not need it to scan: if the admin PC is off, the kiosk shows `Offline - N queued` and delivers the scans when it returns.
* Faculty look at the kiosk camera. The screen shows the name and `CHECK-IN`, `CHECK-OUT`, `CHECK-IN (re-entry)` or `Already recorded`.
* `Not recognised` or `Not sure who this is` means the scan was rejected and nothing was recorded.

### Converting older face data

Face data created by an older version (SVM-based) still works. To convert it, run `python admin/train_lite.py` on the admin PC, or simply register the next person. Kiosks pick up the new file automatically. After conversion, scikit-learn is no longer needed on either machine.

## 9. Fail-Safe Behaviour

| Situation | What happens |
| --- | --- |
| Camera unplugged or not yet detected | The kiosk shows "Camera not available" and keeps retrying until it returns |
| Camera stops delivering frames | The kiosk re-opens the camera automatically |
| Error while processing a frame | That frame is skipped; after repeated errors the models are reloaded |
| Database error while saving a scan | Screen shows "System busy"; the kiosk keeps running |
| Power cut or crash mid-scan | A scan and its upload-queue entry are written in one transaction: both exist or neither does |
| Admin PC off / network down | Scans are queued locally and delivered later, in order, with their real time |
| Reply lost after the admin PC saved a scan | Resent scans are recognised by id and not counted twice |
| Corrupt or tampered classifier download | Rejected before installation; the previous classifier stays active |
| Different model files or library versions on the kiosk | Face data is not installed; a clear message appears on the kiosk and on the Kiosks page |
| Registration fails at any step | Database, photos and face data are restored to their previous state |

Scans are decided on the kiosk, so recognition never waits for the network.

## 10. Configuration

Set these as environment variables before starting the program.

Kiosk (`kiosk/kiosk.py`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `DETECT_EVERY` | 2 | Run face detection on every Nth frame to save CPU |
| `MIN_FACE_PX` | 90 | Ignore faces narrower than this (too far away) |
| `EMB_FRAMES` | 2 | Frames averaged for one recognition |
| `COSINE_THRESHOLD` | 0.363 | Minimum similarity to accept a match |
| `MATCH_MARGIN` | 0.05 | Required lead over the runner-up person |
| `SAME_FACE_MIN` | 0.5 | Frames of one scan must show the same face |
| `RESULT_HOLD` / `REJECT_HOLD` | 3 / 1.5 s | How long a result / a rejection stays on screen |
| `COOLDOWN_SEC` | 60 | Window in which a repeat scan is ignored |
| `COLLEGE_START_TIME` | 10:00:00 | First check-in after this time is marked late |
| `BAD_READ_LIMIT` | 40 | Failed camera reads before the camera is re-opened |
| `CLASSIFIER_CHECK_SEC` | 3 | How often the kiosk checks for new face data |
| `PERF_LOG_SEC` | 30 | Print a timing line every N seconds (0 turns it off) |
| `KIOSK_FULLSCREEN` | 1 | Set to 0 to run in a window |
| `OPENCV_THREADS` | 2 | OpenCV threads on the Pi |
| `KIOSK_CONFIG` | `kiosk_config.json` | Path to the link configuration |

Admin PC (`admin/app.py`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `ATTENDANCE_HOST` / `ATTENDANCE_PORT` | 0.0.0.0 / 5000 | Where the dashboard listens (0.0.0.0 so kiosks can reach it) |
| `ATTENDANCE_PUBLIC_URL` | detected LAN address | Address written into new kiosk configurations |
| `ATTENDANCE_SECRET_KEY` | generated `.secret_key` | Session signing key |
| `ATTENDANCE_SESSION_HOURS` | 8 | Login session length |
| `ATTENDANCE_MAX_FAILED_LOGINS` / `ATTENDANCE_LOCKOUT_SECONDS` | 5 / 900 | Per-address brute-force lockout |
| `ATTENDANCE_COOKIE_SECURE` | 0 | Set to 1 when serving over HTTPS |
| `KIOSK_ONLINE_SECONDS` | 45 | A kiosk counts as online if it contacted the admin PC this recently |

## 11. Performance on the Raspberry Pi 3B+

Three neural networks (YuNet, MiniFAS, SFace) share a small CPU. To see where time goes, the kiosk prints a line every 30 seconds:

```text
[kiosk] [perf] 9.5 fps  detect=40ms embed=55ms frame=95ms spoof=30ms
```

(Example format; your numbers will differ.) Use it to decide what to tune. Options that reduce load: raise `DETECT_EVERY`, raise `MIN_FACE_PX`, or lower `EMB_FRAMES`. Anti-spoofing is part of the security design; weaken it deliberately and only after testing.

Matching itself is cheap and does not slow down as faculty are added (a few hundred people take well under a millisecond).

## 12. Security Notes

* Do not commit `known_faces/`, `attendance.db`, `.secret_key`, `kiosk_config.json` or the `.pkl` face data. Suggested `.gitignore` entries: `admin/.registration/`, `admin/.secret_key`, `admin/attendance.db`, `admin/known_faces/`, `admin/face_classifier_lite.pkl*`, `kiosk/attendance.db`, `kiosk/face_classifier_lite.pkl*`, `kiosk/kiosk_config.json`, `kiosk/.kiosk_state.json`.
* Dashboard pages require an admin login. Kiosk sync requests are authenticated by per-kiosk signatures instead.
* Traffic between kiosk and admin PC is signed but **not encrypted** (plain HTTP). Keep both on a trusted network, or put the dashboard behind HTTPS (and set `ATTENDANCE_COOKIE_SECURE=1`). Faculty names and attendance times travel in clear text.
* Restrict network access to the admin PC's port 5000 to the kiosks and administrators.
* If a kiosk key is exposed, use **New key** (or **Remove**) on the Kiosks page.
* Back up `admin/attendance.db` and `admin/known_faces/` regularly. They are the master data.

## 13. Troubleshooting

| Symptom | Likely cause and fix |
| --- | --- |
| Kiosk shows "No faculty enrolled yet" and "Register faculty from the dashboard first" | No `kiosk_config.json`: the kiosk is stand-alone. Create it from the Kiosks page |
| Same screen with "Waiting for the admin PC" | Linked but nothing received yet. Check the top-right status line, that `admin/app.py` is running, the address in `kiosk_config.json` and the admin PC's firewall |
| Top bar shows "Connecting to admin PC..." | The kiosk cannot reach `admin_url`. Use the admin PC's LAN address (not `127.0.0.1`, unless both run on the same computer) |
| Console says "Wrong kiosk key" | The key does not match. Use **New key** and paste the new configuration |
| Warning about the clock | Fix the kiosk's date and time (enable NTP) |
| Kiosks page says model files differ | Copy the same `common/models/` to the kiosk |
| Registration refused: "too similar to ..." | The new face is too close to an enrolled person for reliable recognition. Nothing was saved |
| `ModuleNotFoundError` for a project file | Run from the repository root (`python admin/app.py`) or from inside the folder; do not copy single files elsewhere |
| Editor underlines `lite_core` imports | Add `"python.analysis.extraPaths": ["./common"]` to `.vscode/settings.json` |
| Camera window does not open when registering | Start `admin/app.py` from the admin PC's desktop session, not as a background service |

Running both parts on one computer for development: use two copies of the project folders (or two data directories), point the kiosk's `admin_url` at `http://127.0.0.1:5000`, set `KIOSK_FULLSCREEN=0`, and close the kiosk while registering, because both want the same webcam.
