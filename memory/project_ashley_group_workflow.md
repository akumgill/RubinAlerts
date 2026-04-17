---
name: MAGNETS collaboration and spectroscopic follow-up workflow
description: MAGNETS collaboration context (pooled Magellan time), Ashley group workflow, and Stubbs 2026B proposal details for Rubin transient follow-up
type: project
---

## MAGNETS Collaboration

MAGNETS = Magellan partner institutions pooling observing time for Rubin transient follow-up. Formed March 2026 with start of LSST alerts.

**Key quote from Stubbs 2026B proposal:**
> "Our collaboration's plan is to pool our awarded observing time and develop an internal queue schedule to address a wide range of science goals... We will apportion the queue observing time and target prioritization in proportion to the time awarded by the respective TACs."

**Your role (from proposal):**
> "Graduate students Akum Gil and Jonah Medoff... will contribute to the collaborative infrastructure needed to execute this blended program and to distribute the data."

## Stubbs 2026B Proposal (propid 2835)

- **Instrument**: LLAMAS integral field spectrograph (Magellan/Baade)
- **Allocation**: 3 nights (0.5D + 2.0G + 0.5B = 30 hours)
- **Semester**: 2026B (Jul 7 - Jan 16)
- **Targets**: 29 (SNe Ia in DDFs + WD standards), r=18-21.5, z=0.1-0.4
- **Queued observing**: Yes
- **Science goal**: Test DESI evolving dark energy hypothesis with high-precision spectrophotometry

## Ashley Group Workflow

Meeting with Yize from Ashley's group (2026-04-14) revealed their current spectroscopic follow-up workflow:

## Current Process

**1. Requesting targets**
- Google Form populates a Google Sheet
- Columns: Timestamp, Name of Requester, Instrument Requested, Target Name (IAUID/ZTFID/ANTARES ID), Brief description, RA (J2000 deg), Dec (J2000 deg), YSE-PZ link (or Alerce/ANTARES), Current Apparent Brightness and Filter, Photometry requested (which bands?), Additional notes, Email Address, Status
- No formal priority field currently, but info used in ranking

**2. Building observing plan**
- Yize manually runs a notebook
- Ad-hoc ranking based on visibility windows and exposure time (from apparent magnitude)
- Produces plan + backup targets for observer flexibility
- Existing script from Alex (for LDSS3) can generate obs plans but not currently used

**3. Output data**
- Raw data → Google Drive → processed → reduced data to SkyPortal instance

## Potential Integration Points

- Orchestration layer between target requests and obs plan generation
- Two-parameter prioritization: requested priority + available observing time
- Time accounting system needed per Chris

**Why:** Chris indicated their coordinated proposal "needs to be much more automated including some kind of time accounting" - current manual process won't scale with more requesters.

**How to apply:** When designing alert→follow-up pipeline, consider:
- Compatibility with their Google Sheet schema for target requests
- Integration with existing LDSS3 obs plan script as starting point
- Time accounting as a first-class feature
- SkyPortal as downstream data destination

## Resources to obtain
- Alex's LDSS3 obs plan generation script (Dropbox - needs download)
- Example generated observing plan (Dropbox - needs download)
