# Onshape API keys (for the `onshape / native` route)

The router reads your real Onshape mates through the Onshape REST API, using cad2urdf's own client (`cad2urdf/onshape.py`). That needs an API key pair.

API keys are managed in your Onshape **settings** (the old "Developer portal" link no longer shows them). Where depends on your account type:

1. **Personal / free / education account:** user icon (top right) → **My account** → **Developer** → **API keys** → **Create new API key**. Tick read permissions only.
2. **Company or Enterprise account** (e.g. `tritonrobotics.onshape.com`): **only an admin can create keys** for Enterprise documents. Ask an admin to do:
   - user icon → **Company/Enterprise settings** → **Developer** → **API keys** tab → **Create new API key**;
   - assign the key to you, with read permissions;
   - send you both values privately.

   The *My account → Developer* page only manages keys not tied to the company; those may not be able to read Enterprise documents.
3. Copy the **access key** and the **secret key**. The secret is shown only once.

   Sources: [My Account – Developer](https://cad.onshape.com/help/Content/Plans/my_account_developer.htm), [Company/Classroom/Enterprise Settings – Developer](https://cad.onshape.com/help/Content/Plans/enterprise_settings_developer.htm).

**No admin available?** Use Onshape's URDF export instead (no keys needed; see the end of this page).

4. Put them in your shell. Never commit them or paste them into chat:

   ```bash
   export ONSHAPE_API=https://tritonrobotics.onshape.com   # your domain; cad.onshape.com for normal accounts
   export ONSHAPE_ACCESS_KEY=<access key>
   export ONSHAPE_SECRET_KEY=<secret key>
   ```

   Or put the same two lines (without `export`) in a **`.env` file at the repo root**. It is git-ignored, and cad2urdf reads it automatically:

   ```
   ONSHAPE_ACCESS_KEY=<access key>
   ONSHAPE_SECRET_KEY=<secret key>
   ```

5. Run the route with the assembly's URL (the `.../e/<element id>` part must be the **assembly** tab):

   ```bash
   python -m cad2urdf.route --cad onshape --format native --sim maniskill --run \
       --input "https://tritonrobotics.onshape.com/documents/<doc>/w/<workspace>/e/<assembly>" --out build/hero
   ```

## What your assembly needs for this route

Nothing special. Every mate is read as it is:
- fastened mates and rigid sub-assemblies merge into one link;
- revolute and slider mates become joints;
- gear-type relations become mimic joints;
- mate limits become joint limits.

The instance marked *Fixed* (or the heaviest group) is the base. Assign materials in Onshape so the masses are real; parts without a material get `default_density` from the spec.

Without keys, you can use Onshape's built-in URDF export instead: right-click the assembly tab → **Export** → **URDF**, then:

```bash
python -m cad2urdf.route --cad onshape --format urdf-export --sim maniskill --run --input <unzipped>/robot.urdf --out build/hero
```
