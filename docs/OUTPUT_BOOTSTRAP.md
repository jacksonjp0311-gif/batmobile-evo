# BATMOBILE.EVO — Output Bootstrap Technique

Every All-One execution MUST emit:

1. logs/run_<TAG>.log
   → complete console truth of the run

2. artifacts/latest_<TAG>.txt
   → pointer to the newest benchmark artifact JSON

Purpose:
- Future iterations can be shared by file link, not pasted terminal text.
- Run truth becomes reproducible + auditable.