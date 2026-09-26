# Onshape API keys (for the `onshape / native` route)

The router reads your real Onshape mates through the REST API (via onshape-to-robot). That needs an API key pair.

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

onshape-to-robot turns only **mate connectors named `dof_<joint>`** into joints: revolute/cylindrical become revolute, slider becomes prismatic. Everything else is fixed. The first instance in the assembly (or the one marked *Fixed*) is the base.

If your assembly doesn't use that naming, use Onshape's built-in URDF export instead. It needs no keys, and every mate becomes a joint: right-click the assembly tab → **Export** → **URDF**, then:

```bash
python -m cad2urdf.route --cad onshape --format urdf-export --sim maniskill --run --input <unzipped>/robot.urdf --out build/hero
```
