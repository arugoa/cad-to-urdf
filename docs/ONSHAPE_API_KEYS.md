# Onshape API keys (for the `onshape / native` route)

The router reads your real Onshape mates through the Onshape REST API, using cad2urdf's own client (`cad2urdf/onshape.py`). That needs an API key pair.

1. Sign in to Onshape with your team account, then open **https://dev-portal.onshape.com/keys**. You can also get there from Onshape: profile icon (top right) → **Developer portal** → **API keys**.
2. Click **Create new API key**. Tick the read permissions only: *read your profile information* and *read your documents*.
   - Enterprise accounts (e.g. `tritonrobotics.onshape.com`): if the option isn't there, an Enterprise admin has to allow API keys for the domain.
3. Copy the **access key** and the **secret key**. The secret is shown only once.
4. Put them in your shell. Never commit them or paste them into chat:

   ```bash
   export ONSHAPE_API=https://tritonrobotics.onshape.com   # your domain; cad.onshape.com for normal accounts
   export ONSHAPE_ACCESS_KEY=<access key>
   export ONSHAPE_SECRET_KEY=<secret key>
   ```

   To keep them across sessions, put the lines in `~/.bashrc` or in an untracked `.env` file that you `source`.

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
