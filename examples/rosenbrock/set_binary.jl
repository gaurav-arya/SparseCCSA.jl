using Preferences
using NLopt_jll

# need to restart Julia after setting preferences

set_preferences!(
    NLopt_jll,
    "libnlopt_path" => "libnlopt"
)

using NLopt_jll

NLopt_jll.get_libnlopt_path()



