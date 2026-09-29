# Onshape API keys

The `onshape / native` route reads your assembly through the Onshape REST API and needs an API key pair. Keys are managed in your Onshape settings:

- Personal, free or education account: user icon (top right), My account, Developer, API keys, Create new API key. Read permissions are enough.
- Company or Enterprise account (e.g. `yourteam.onshape.com`): only an admin can create keys that read Enterprise documents. The admin goes to user icon, Enterprise settings, Developer, API keys, creates a read-only key assigned to you, and sends you both values. A personal key gets a 403 on Enterprise documents.

The secret key is shown only once. Onshape's help pages: [My Account – Developer](https://cad.onshape.com/help/Content/Plans/my_account_developer.htm), [Enterprise Settings – Developer](https://cad.onshape.com/help/Content/Plans/enterprise_settings_developer.htm).

Put the keys in `.env` at the repo root (git-ignored, read automatically; exported shell variables take precedence):

```bash
cp .env.example .env
# then set
ONSHAPE_ACCESS_KEY=<access key>
ONSHAPE_SECRET_KEY=<secret key>
```

Run the router with the URL of the assembly tab (the `/e/<id>` part must be the assembly):

```bash
python -m cad2urdf.route --cad onshape --format native --sim maniskill --run \
    --input "https://cad.onshape.com/documents/<doc>/w/<workspace>/e/<assembly>" --out build/robot
```

Every mate is read as it is, with no naming convention required. The instance marked Fixed (or the heaviest group) is the base. Assign materials in Onshape so the masses are real; parts without one get `default_density` from the spec.

No keys? Use Onshape's URDF export instead: right-click the assembly tab, Export, URDF, then

```bash
python -m cad2urdf.route --cad onshape --format urdf-export --sim maniskill --run --input <unzipped>/robot.urdf --out build/robot
```
