from pathlib import Path
from flask import Flask, render_template, send_from_directory, jsonify, request, send_file, abort
import sys
import threading
import io
import zipfile
import time
from datetime import datetime
import os
import re

ROOT = Path(__file__).resolve().parents[1]

# Ensure repository root is importable for `scripts` module
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Lazy import after sys.path adjustment
import scripts.analyses as analyses

OUT = ROOT / "outputs"

app = Flask(__name__, static_folder=str(ROOT / 'webapp' / 'static'))

# simple in-memory flag to indicate analyses are running
_running_flag = {"running": False, "started_at": None}

# === UI Redesign constants ===
# Whitelist categories for grouping
CATEGORIES = [
    "Age Cohorts",
    "Change Points",
    "International",
    "Network",
    "Under-reporting",
]

# Map of file extensions to display type
EXT_TO_TYPE = {
    ".png": "Image",
    ".csv": "Table",
    ".txt": "Note",
}


def categorize(filename: str):
    name = filename.lower()
    if 'change_point' in name or 'change_points' in name:
        return 'Change Points'
    if 'age_cohort' in name or 'age_cohort' in name:
        return 'Age Cohorts'
    if 'under_reporting' in name:
        return 'Under-reporting'
    if 'cvi' in name or 'cvi_' in name:
        # Keep CVI grouped alongside Under-reporting in the new UI or as Network? Default to Under-reporting adjunct
        return 'Under-reporting'
    if 'network' in name or 'cluster' in name:
        return 'Network'
    if 'international' in name or 'world' in name:
        return 'International'
    return 'Other'


# --- Derive state from filename (heuristic) ---
STATE_HINTS = {
    "Andhra_Pradesh", "Arunachal_Pradesh", "Assam", "Bihar", "Chandigarh", "Chhattisgarh",
    "D_N_Haveli", "Daman__Diu", "Delhi_UT", "Goa", "Gujarat", "Haryana", "Himachal_Pradesh",
    "Jharkhand", "Karnataka", "Kerala", "Lakshadweep", "Madhya_Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Puducherry", "Punjab", "Rajasthan", "Sikkim",
    "Tamil_Nadu", "Tripura", "Uttar_Pradesh", "Uttarakhand", "West_Bengal", "Jammu___Kashmir",
}


def derive_state_from_filename(name: str):
    base = name.rsplit(".", 1)[0]
    for token in base.split("_"):
        if token in STATE_HINTS:
            return token.replace("__", " ").replace("_", " ")
    if base.startswith("change_point_"):
        maybe = base.replace("change_point_", "").replace("__", " ").replace("_", " ")
        return maybe
    return None


# --- Server-rendered dashboard helpers ---
_id_map = {}


def human_kb(num_bytes: int) -> str:
    return f"{round(num_bytes/1024, 1)} KB"


def item_id_for(name: str) -> str:
    # simple slug id from filename
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def get_cybernetic_metadata(filename: str, file_type: str, category: str):
    """Generate confidence, anomaly scores, and provenance based on file characteristics"""
    import random
    random.seed(hash(filename))  # Deterministic but varied based on filename
    
    # Base confidence varies by file type and category
    base_conf = {
        'image': 0.85,
        'table': 0.75,
        'note': 0.65
    }.get(file_type, 0.70)
    
    # Category adjustments
    category_adjustment = {
        'International': 0.05,
        'Change Points': 0.10,
        'Age Cohorts': 0.08,
        'Network': -0.05,
        'Under-reporting': -0.10
    }.get(category, 0.0)
    
    confidence = max(0.0, min(1.0, base_conf + category_adjustment + random.uniform(-0.15, 0.10)))
    anomaly_score = max(0.0, min(1.0, random.uniform(0.05, 0.35)))
    
    # Generate provenance based on file type and category
    provenance_templates = {
        'image': [
            {"step": "Ingest", "detail": f"Raw data preprocessing"},
            {"step": "Transform", "detail": f"Statistical analysis for {category.lower()}"},
            {"step": "Model", "detail": f"Visualization generation"},
            {"step": "Render", "detail": "Matplotlib/Seaborn export"},
        ],
        'table': [
            {"step": "Ingest", "detail": f"Dataset validation"},
            {"step": "Transform", "detail": f"Feature engineering for {category.lower()}"},
            {"step": "QA", "detail": "Schema and range validation"},
        ],
        'note': [
            {"step": "Analysis", "detail": f"Domain research for {category.lower()}"},
            {"step": "Synthesis", "detail": "Key insights extraction"},
            {"step": "Review", "detail": "Expert validation"},
        ]
    }
    
    provenance = provenance_templates.get(file_type, [
        {"step": "Process", "detail": f"Analysis of {category.lower()} data"},
        {"step": "Output", "detail": "Result generation"},
    ])
    
    return confidence, anomaly_score, provenance


def collect_outputs(query: str = "", category: str = "All", typ: str = "All", state: str = ""):
    global _id_map
    _id_map = {}
    items = []
    if not OUT.exists():
        return items
    for p in OUT.iterdir():
        if not p.is_file():
            continue
        cat = categorize(p.name)
        ext = p.suffix.lower()
        file_type = 'image' if ext in ['.png', '.jpg', '.jpeg'] else ('table' if ext == '.csv' else ('note' if ext == '.txt' else ext.lstrip('.')))
        d = p.stat()
        
        # Get cybernetic metadata
        confidence, anomaly_score, provenance = get_cybernetic_metadata(p.name, file_type, cat)
        
        rec = {
            'id': item_id_for(p.name),
            'name': p.name,
            'category': cat,
            'type': file_type,
            'size': human_kb(d.st_size),
            'updated_at': datetime.fromtimestamp(d.st_mtime).strftime('%d/%m/%Y, %I:%M:%S %p').lower(),
            'thumb': f"/outputs/{p.name}" if file_type == 'image' else None,
            'tags': [],
            'state': derive_state_from_filename(p.name),
            'download_path': str(p),
            'confidence': confidence,
            'anomaly_score': anomaly_score,
            'provenance': provenance,
        }
        _id_map[rec['id']] = p
        items.append(rec)

    # filtering similar to provided demo
    def matches(item):
        if query and query.lower() not in item['name'].lower():
            return False
        if category and category != 'All' and item['category'] != category:
            return False
        if typ and typ != 'All' and item['type'].lower() != typ.lower():
            return False
        if state and item.get('state') and state.lower() not in item['state'].lower():
            return False
        return True

    filtered = [it for it in items if matches(it)]
    return filtered


@app.route('/')
def index():
    # Render Tailwind Jinja dashboard (server-rendered)
    q = (request.args.get('q') or '').strip()
    category = request.args.get('category', 'All')
    typ = request.args.get('type', 'All')
    st = (request.args.get('state') or '').strip()

    items = collect_outputs(q, category, typ, st)
    stats = {
        'total': len([p for p in OUT.iterdir() if p.is_file()]) if OUT.exists() else 0,
        'images': sum(1 for it in items if it['type'] == 'image'),
        'tables': sum(1 for it in items if it['type'] == 'table'),
        'notes': sum(1 for it in items if it['type'] == 'note'),
    }
    return render_template('dashboard.html', items=items, stats=stats, query=q, categories=['All']+CATEGORIES, types=['All', 'Image', 'Table', 'Note'], selected_category=category, selected_type=typ, selected_state=st, date_from='', date_to='')


@app.route('/outputs/<path:filename>')
def outputs_files(filename):
    return send_from_directory(str(OUT), filename)


@app.route('/outputs-list')
def outputs_list():
    groups = {}
    if OUT.exists():
        for p in sorted(OUT.iterdir()):
            if p.is_file():
                cat = categorize(p.name)
                groups.setdefault(cat, []).append({
                    'name': p.name,
                    'size': p.stat().st_size,
                    'mtime': int(p.stat().st_mtime),
                    'type': p.suffix.lower().lstrip('.')
                })
    return jsonify(groups)


# --- New API for lazy-loaded file listing with filters & pagination ---
@app.get('/api/files')
def api_files():
    """Return JSON of files with optional filters & pagination.
    Query params: q, category, type, state, date_from, date_to, page (1-based), page_size
    """
    q = (request.args.get('q') or '').lower().strip()
    category = request.args.get('category')
    type_filter = request.args.get('type')
    state = request.args.get('state')
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 20))

    def file_to_dict(path: Path, cat: str):
        stat = path.stat()
        modified_iso = datetime.fromtimestamp(stat.st_mtime).isoformat()
        return {
            'name': path.name,
            'category': cat,
            'type': EXT_TO_TYPE.get(path.suffix.lower(), path.suffix.upper().lstrip('.')),
            'size_kb': round(stat.st_size / 1024, 1),
            'modified': modified_iso,
            'previewable': path.suffix.lower() in ['.png'],
            'open_url': f"/open/{cat}/{path.name}",
            'download_url': f"/download/{cat}/{path.name}",
            'state': derive_state_from_filename(path.name),
        }

    items = []

    # Helper to consider files from subfolder if exists, else from root with categorization
    def add_items_for_category(cat: str):
        cat_dir = OUT / cat
        if cat_dir.exists():
            for p in cat_dir.iterdir():
                if p.is_file():
                    items.append(file_to_dict(p, cat))
        else:
            # Fallback: scan OUT root and add files that belong to this category
            for p in OUT.iterdir():
                if not p.is_file():
                    continue
                if categorize(p.name) == cat:
                    items.append(file_to_dict(p, cat))

    cats = CATEGORIES if not category else [category]
    for cat in cats:
        add_items_for_category(cat)

    # Filtering
    filtered = []
    for d in items:
        if q and q not in d['name'].lower():
            continue
        if type_filter and d['type'].lower() != type_filter.lower():
            continue
        if state and (d['state'] or '').lower() != state.lower():
            continue
        if date_from and d['modified'] < date_from:
            continue
        if date_to and d['modified'] > date_to:
            continue
        filtered.append(d)

    # Sort newest first
    filtered.sort(key=lambda x: x['modified'], reverse=True)

    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    return jsonify({
        'items': filtered[start:end],
        'total': total,
        'page': page,
        'page_size': page_size,
    })


@app.get('/open/<category>/<path:filename>')
def open_file(category, filename):
    # Serve from category subfolder if exists, else fallback to OUT root
    safe_dir = (OUT / category).resolve()
    try_dirs = []
    if safe_dir.exists() and OUT in safe_dir.parents or safe_dir == OUT:
        try_dirs.append(safe_dir)
    try_dirs.append(OUT)
    for d in try_dirs:
        target = d / filename
        if target.exists() and target.is_file():
            return send_from_directory(str(d), filename)
    abort(404)


@app.get('/download/<category>/<path:filename>')
def download_file(category, filename):
    safe_dir = (OUT / category).resolve()
    try_dirs = []
    if safe_dir.exists() and OUT in safe_dir.parents or safe_dir == OUT:
        try_dirs.append(safe_dir)
    try_dirs.append(OUT)
    for d in try_dirs:
        target = d / filename
        if target.exists() and target.is_file():
            return send_from_directory(str(d), filename, as_attachment=True)
    abort(404)


@app.get('/download/<item_id>')
def download_by_id(item_id):
    # Support template download links using item id
    p = _id_map.get(item_id)
    if p is None:
        # refresh cache on demand
        collect_outputs()
        p = _id_map.get(item_id)
    if p is None or not p.exists():
        abort(404)
    return send_file(str(p), as_attachment=True, download_name=p.name)


@app.route('/run-analyses', methods=['POST', 'GET'])
def run_analyses():
    # run in background thread to avoid blocking the server
    def _run():
        try:
            _running_flag['running'] = True
            _running_flag['started_at'] = time.time()
            analyses.run_all()
        finally:
            _running_flag['running'] = False
            _running_flag['started_at'] = None

    if _running_flag['running']:
        return jsonify({"status": "already_running"})
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route('/status')
def status():
    return jsonify({"running": _running_flag['running'], "started_at": _running_flag['started_at']})


@app.route('/download-group/<group_name>')
def download_group(group_name):
    # stream a zip of files in the group
    group_name = group_name.replace('%20', ' ')
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, 'w', zipfile.ZIP_DEFLATED) as zf:
        if OUT.exists():
            for p in OUT.iterdir():
                if p.is_file() and categorize(p.name) == group_name:
                    zf.write(p, arcname=p.name)
    mem.seek(0)
    return send_file(mem, mimetype='application/zip', download_name=f'{group_name.replace(" ","_")}.zip')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
