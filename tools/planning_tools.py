"""Planning and task management tools for structured work breakdown."""

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

from tools.registry import ToolDef, TOOL_REGISTRY

PLANS_DIR = Path(__file__).parent.parent / 'plans'
PLANS_DIR.mkdir(exist_ok=True)

VALID_STATUSES = ('pending', 'in_progress', 'completed', 'blocked')


def _current_plan_path() -> Path:
    return PLANS_DIR / '_current.json'


def _load_plan(name: str | None = None) -> dict | None:
    if name:
        path = PLANS_DIR / f'{name}.json'
    else:
        path = _current_plan_path()
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _save_plan(plan: dict, name: str | None = None):
    plan['updated'] = datetime.now(timezone.utc).isoformat()
    if name:
        path = PLANS_DIR / f'{name}.json'
    else:
        path = _current_plan_path()
    path.write_text(json.dumps(plan, indent=2))


def register():
    """Register planning tools with the global registry."""

    async def create_plan(name: str, goal: str, tasks: str) -> dict:
        """Create a new plan. tasks is a pipe-separated list of task descriptions."""
        safe_name = ''.join(c for c in name if c.isalnum() or c in '-_').lower()
        if not safe_name:
            return {'error': 'Invalid plan name'}

        task_list = [t.strip() for t in tasks.split('|') if t.strip()]
        if not task_list:
            return {'error': 'No tasks provided. Separate tasks with | characters.'}

        plan = {
            'name': safe_name,
            'goal': goal,
            'created': datetime.now(timezone.utc).isoformat(),
            'updated': datetime.now(timezone.utc).isoformat(),
            'tasks': [
                {'id': i + 1, 'description': desc, 'status': 'pending', 'notes': ''}
                for i, desc in enumerate(task_list)
            ],
        }

        _save_plan(plan, safe_name)
        _save_plan(plan)  # also set as current
        return {'message': f'Plan "{safe_name}" created with {len(task_list)} tasks', 'plan': plan}

    async def get_plan(name: str = '') -> dict:
        """Get a plan by name, or the current plan if no name given."""
        plan = _load_plan(name if name else None)
        if not plan:
            label = f'Plan "{name}"' if name else 'No current plan'
            return {'error': f'{label} not found'}
        return {'plan': plan}

    async def update_task(task_id: str, status: str, notes: str = '') -> dict:
        """Update a task's status in the current plan."""
        plan = _load_plan()
        if not plan:
            return {'error': 'No current plan. Create one first with create_plan.'}

        if status not in VALID_STATUSES:
            return {'error': f'Invalid status "{status}". Must be one of: {", ".join(VALID_STATUSES)}'}

        tid = int(task_id)
        for task in plan['tasks']:
            if task['id'] == tid:
                task['status'] = status
                if notes:
                    task['notes'] = notes
                _save_plan(plan)
                _save_plan(plan, plan['name'])
                return {'message': f'Task {tid} updated to {status}', 'task': task}

        return {'error': f'Task {tid} not found in plan'}

    async def add_task(description: str, after_task_id: str = '') -> dict:
        """Add a task to the current plan."""
        plan = _load_plan()
        if not plan:
            return {'error': 'No current plan. Create one first with create_plan.'}

        max_id = max(t['id'] for t in plan['tasks']) if plan['tasks'] else 0
        new_task = {'id': max_id + 1, 'description': description, 'status': 'pending', 'notes': ''}

        if after_task_id:
            idx = next((i for i, t in enumerate(plan['tasks']) if t['id'] == int(after_task_id)), None)
            if idx is not None:
                plan['tasks'].insert(idx + 1, new_task)
            else:
                plan['tasks'].append(new_task)
        else:
            plan['tasks'].append(new_task)

        _save_plan(plan)
        _save_plan(plan, plan['name'])
        return {'message': f'Task {new_task["id"]} added: {description}', 'task': new_task}

    async def list_plans() -> dict:
        """List all saved plans."""
        plans = []
        for f in sorted(PLANS_DIR.glob('*.json')):
            if f.name == '_current.json':
                continue
            try:
                p = json.loads(f.read_text())
                total = len(p.get('tasks', []))
                done = sum(1 for t in p.get('tasks', []) if t['status'] == 'completed')
                plans.append({
                    'name': p.get('name', f.stem),
                    'goal': p.get('goal', ''),
                    'progress': f'{done}/{total}',
                    'updated': p.get('updated', ''),
                })
            except Exception:
                continue
        if not plans:
            return {'message': 'No plans found'}
        return {'plans': plans}

    # Register all tools
    TOOL_REGISTRY.register(ToolDef(
        name='create_plan',
        description='Create a structured plan with a goal and task list. Use this when starting any non-trivial task to organize your work.',
        parameters={
            'name': {'type': 'string', 'description': 'Short plan name (alphanumeric, hyphens, underscores)'},
            'goal': {'type': 'string', 'description': 'What this plan aims to accomplish'},
            'tasks': {'type': 'string', 'description': 'Tasks separated by the pipe character |. Each segment between pipes becomes ONE separate task. Example: "Research the topic | Create an outline | Write the first draft | Review and edit". Do NOT put multiple steps in a single segment. Break work into small, concrete tasks.'},
        },
        handler=create_plan,
    ))

    TOOL_REGISTRY.register(ToolDef(
        name='get_plan',
        description='View the current plan or a named plan. Shows all tasks with their status.',
        parameters={
            'name': {'type': 'string', 'description': 'Plan name to view (empty for current plan)'},
        },
        required=[],
        handler=get_plan,
    ))

    TOOL_REGISTRY.register(ToolDef(
        name='update_task',
        description='Update a task status in the current plan. Mark tasks as you work through them.',
        parameters={
            'task_id': {'type': 'string', 'description': 'Task ID number'},
            'status': {'type': 'string', 'description': 'New status: pending, in_progress, completed, or blocked'},
            'notes': {'type': 'string', 'description': 'Optional notes about progress or blockers'},
        },
        required=['task_id', 'status'],
        handler=update_task,
    ))

    TOOL_REGISTRY.register(ToolDef(
        name='add_task',
        description='Add a new task to the current plan. Use when you discover additional work needed.',
        parameters={
            'description': {'type': 'string', 'description': 'Task description'},
            'after_task_id': {'type': 'string', 'description': 'Insert after this task ID (empty to append)'},
        },
        required=['description'],
        handler=add_task,
    ))

    TOOL_REGISTRY.register(ToolDef(
        name='list_plans',
        description='List all saved plans with their progress.',
        parameters={},
        required=[],
        handler=list_plans,
    ))
