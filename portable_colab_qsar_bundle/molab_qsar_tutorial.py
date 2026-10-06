import marimo



__generated_with = "0.0.0"

app = marimo.App(width="full")



@app.cell
def _():
    import sys
    import importlib.util
    from pathlib import Path
    from urllib.error import URLError
    from urllib.request import urlretrieve
    import marimo as mo

    _here = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
    _candidates = [_here, _here.parent, Path.cwd(), Path.cwd().parent, Path('/mnt/data'), Path('/tmp')]
    for _candidate in _candidates:
        if (_candidate / 'portable_colab_qsar_bundle').exists() and str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))

    def _load_qsarena_molab_runtime():
        try:
            from portable_colab_qsar_bundle import molab_qsar_runtime
            return molab_qsar_runtime
        except ModuleNotFoundError:
            pass
        sibling_candidates = [
            _here / 'molab_qsar_runtime.py',
            Path.cwd() / 'molab_qsar_runtime.py',
            _here / 'portable_colab_qsar_bundle' / 'molab_qsar_runtime.py',
            Path.cwd() / 'portable_colab_qsar_bundle' / 'molab_qsar_runtime.py',
        ]
        for runtime_path in sibling_candidates:
            if runtime_path.exists():
                spec = importlib.util.spec_from_file_location('qsarena_molab_runtime', runtime_path)
                module = importlib.util.module_from_spec(spec)
                assert spec is not None and spec.loader is not None
                spec.loader.exec_module(module)
                return module
        cache_dir = Path.cwd() / '.qsarena_molab'
        cache_dir.mkdir(parents=True, exist_ok=True)
        runtime_path = cache_dir / 'molab_qsar_runtime.py'
        runtime_urls = [
            'https://raw.githubusercontent.com/ScottCoffin/QSARena/main/portable_colab_qsar_bundle/molab_qsar_runtime.py',
            'https://raw.githubusercontent.com/ScottCoffin/AutoQSAR/main/portable_colab_qsar_bundle/molab_qsar_runtime.py',
        ]
        errors = []
        for url in runtime_urls:
            try:
                urlretrieve(url, runtime_path)
                print(f'[QSARena Molab] downloaded runtime support module from {url}', flush=True)
                break
            except Exception as exc:
                errors.append(f'{url}: {type(exc).__name__}: {exc}')
        else:
            raise ModuleNotFoundError(
                'Could not locate molab_qsar_runtime.py. Open this notebook from the QSARena GitHub repo, '
                'upload molab_qsar_runtime.py beside it, or push the generated runtime file so Molab can download it. '
                + ' | '.join(errors)
            )
        spec = importlib.util.spec_from_file_location('qsarena_molab_runtime', runtime_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        return module

    try:
        qsarena_molab_runtime = _load_qsarena_molab_runtime()
        RUNTIME_LOAD_ERROR = ''
    except Exception as exc:
        class _MissingRuntime:
            MARKDOWN = []
            CONTROLS = []
            CODE = []
            MANIFEST = []
            @staticmethod
            def run_code(index, namespace):
                raise RuntimeError(RUNTIME_LOAD_ERROR)
        qsarena_molab_runtime = _MissingRuntime()
        RUNTIME_LOAD_ERROR = str(exc)

    def _molab_widget_from_schema(item):
        name = item.get('name', '')
        label = item.get('label') or name.replace('_', ' ')
        kind = item.get('widget_kind', 'string')
        default = item.get('default')
        if kind == 'choice':
            options = list(item.get('options') or [])
            value = default if default in options else (options[0] if options else None)
            return mo.ui.dropdown(options=options, value=value, label=label)
        if kind == 'boolean':
            return mo.ui.checkbox(value=bool(default), label=label)
        if kind == 'slider':
            kwargs = {'value': default, 'label': label, 'show_value': True}
            if 'min' in item:
                kwargs['start'] = item['min']
            if 'max' in item:
                kwargs['stop'] = item['max']
            if 'step' in item:
                kwargs['step'] = item['step']
            return mo.ui.slider(**kwargs)
        if kind in {'integer', 'number'}:
            kwargs = {'value': default, 'label': label}
            if 'min' in item:
                kwargs['start'] = item['min']
            if 'max' in item:
                kwargs['stop'] = item['max']
            if 'step' in item:
                kwargs['step'] = item['step']
            elif kind == 'integer':
                kwargs['step'] = 1
            return mo.ui.number(**kwargs)
        return mo.ui.text(value='' if default is None else str(default), label=label, full_width=True)

    def make_molab_form(title, schema):
        widgets = {}
        for item in schema:
            if item.get('widget_kind') == 'markdown':
                continue
            name = item.get('name')
            if name:
                widgets[name] = _molab_widget_from_schema(item)
        return mo.ui.dictionary(widgets)

    def read_molab_form_values(form, schema):
        values = {}
        for item in schema:
            name = item.get('name')
            if not name:
                continue
            try:
                values[name] = form[name].value
            except Exception:
                values[name] = item.get('default')
        return values

    globals()['MOLAB_RUNTIME'] = True
    globals()['make_molab_form'] = make_molab_form
    globals()['read_molab_form_values'] = read_molab_form_values
    globals()['mo'] = mo
    qsarena_molab_cell_0000_done = True
    return MOLAB_RUNTIME, RUNTIME_LOAD_ERROR, make_molab_form, mo, qsarena_molab_runtime, read_molab_form_values, qsarena_molab_cell_0000_done

@app.cell
def _(RUNTIME_LOAD_ERROR, mo, qsarena_molab_runtime, qsarena_molab_cell_0000_done):
    def _molab_code_title(source, fallback):
        first = source.splitlines()[0].strip() if source.splitlines() else ''
        return first[1:].strip() if first.startswith('#') else fallback

    if RUNTIME_LOAD_ERROR:
        code_titles = ['Runtime support module missing']
    else:
        code_titles = [
            f'{idx:02d}. ' + _molab_code_title(source, f'Step {idx}')
            for idx, source in enumerate(qsarena_molab_runtime.CODE)
        ]
    step_options = {title: idx for idx, title in enumerate(code_titles)}
    step_selector = mo.ui.dropdown(
        options=step_options,
        value=code_titles[0] if code_titles else None,
        label='Runnable tutorial block',
        searchable=True,
        full_width=True,
    )
    run_selected_step = mo.ui.run_button(label='Run selected block', kind='success')
    qsarena_molab_cell_0001_done = True
    runtime_status = (
        '**Runtime support module missing.** Push `portable_colab_qsar_bundle/molab_qsar_runtime.py` to GitHub or upload it beside this notebook, then rerun this cell.\n\n' + RUNTIME_LOAD_ERROR
        if RUNTIME_LOAD_ERROR else
        'Runtime support module loaded.'
    )
    mo.vstack([
        mo.md('# QSARena Molab Tutorial'),
        mo.md('Choose a runnable block, adjust its controls, then press **Run selected block**. Run blocks from top to bottom for a normal tutorial pass.'),
        mo.md(runtime_status),
        step_selector,
        run_selected_step,
    ])
    return code_titles, step_options, run_selected_step, step_selector, qsarena_molab_cell_0001_done

@app.cell
def _(make_molab_form, mo, RUNTIME_LOAD_ERROR, qsarena_molab_runtime, qsarena_molab_cell_0001_done, step_selector, step_options):
    selected_code_index = int(step_options.get(step_selector.value, 0))
    control_payload = None
    active_form_var = None
    if not RUNTIME_LOAD_ERROR:
        for pos, item in enumerate(qsarena_molab_runtime.MANIFEST):
            if item.get('kind') == 'code' and int(item.get('index')) == selected_code_index:
                active_form_var = item.get('form_var')
                if pos > 0 and qsarena_molab_runtime.MANIFEST[pos - 1].get('kind') == 'control':
                    control_payload = qsarena_molab_runtime.CONTROLS[int(qsarena_molab_runtime.MANIFEST[pos - 1]['index'])]
                break
    if RUNTIME_LOAD_ERROR:
        current_form = mo.ui.dictionary({})
        control_title = 'Runtime support module missing'
        control_schema = []
    elif control_payload is None:
        current_form = mo.ui.dictionary({})
        control_title = 'No controls for this block'
        control_schema = []
    else:
        control_title = control_payload['title']
        control_schema = control_payload['schema']
        current_form = make_molab_form(control_title, control_schema)
    globals()['qsarena_molab_active_form'] = current_form
    globals()['qsarena_molab_active_schema'] = control_schema
    if active_form_var:
        globals()[active_form_var] = current_form
    qsarena_molab_cell_0002_done = True
    mo.vstack([mo.md('### Controls: ' + control_title), current_form])
    return control_schema, current_form, active_form_var, selected_code_index, qsarena_molab_cell_0002_done

@app.cell
def _(control_schema, current_form, active_form_var, RUNTIME_LOAD_ERROR, qsarena_molab_runtime, qsarena_molab_cell_0002_done, read_molab_form_values, run_selected_step, selected_code_index):
    globals()['read_molab_form_values'] = read_molab_form_values
    globals()['qsarena_molab_active_form'] = current_form
    globals()['qsarena_molab_active_schema'] = control_schema
    if active_form_var:
        globals()[active_form_var] = current_form
    if RUNTIME_LOAD_ERROR:
        mo.md('Runtime support module is missing; the selected block cannot run yet.')
    elif run_selected_step.value:
        qsarena_molab_runtime.run_code(selected_code_index, globals())
    else:
        mo.md('Press **Run selected block** to execute the selected tutorial block.')
    qsarena_molab_cell_0003_done = True
    return qsarena_molab_cell_0003_done,



if __name__ == "__main__":

    app.run()

