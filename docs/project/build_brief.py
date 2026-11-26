"""Generate the two-page brief from frozen measured results (ReportLab required)."""
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
ROOT=Path(__file__).resolve().parents[2]
r=json.loads((ROOT/'docs/project/FINAL_RESULTS.json').read_text())
out=ROOT/'output/pdf';out.mkdir(parents=True,exist_ok=True)
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Body',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=9,textColor=colors.HexColor('#263449')))
styles['Title'].textColor=colors.HexColor('#10243a');styles['Title'].fontSize=26
styles['Heading2'].textColor=colors.HexColor('#126c71');styles['Heading2'].spaceBefore=12
story=[]
def p(text):story.append(Paragraph(text,styles['Body']))
def h(text):story.append(Paragraph(text,styles['Heading2']))
def table(data):
 t=Table(data,colWidths=[155,115,115,115]);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#10243a')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),('LEADING',(0,0),(-1,-1),12),('BOTTOMPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),9),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#edf5f5'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP')]));story.append(t)
story.append(Paragraph('Can It Play DOOM?',styles['Title']))
p('<b>Yugaantar2.0 | Sensor-only reinforcement learning</b><br/>Custom PPO, a compact neural policy, and measured multi-task evaluation.')
h('What we built')
p('A single offline PyTorch policy turns the official spatial sensors into nine-button actions. ViZDoom runs the game; the unchanged organizer doom_env supplies observations and applies each action for four ticks. The agent learns through practice episodes, with no memorized action sequence or scripted steering.')
h('Architecture')
p('<b>64 ray distances + 64 x 6 hit-type features + 17 vitals = 465 inputs.</b><br/>Two 128-unit tanh layers feed nine Bernoulli action logits and a scalar value head. The network has 77,450 parameters. The value head helps PPO during training; inference samples the action head. Weights load once; reset clears episode state/RNG. No network, map IDs, absolute coordinates or training counters enter the policy.')
h('Learning pipeline')
p('Custom PPO uses 512-decision rollouts, four update epochs, minibatches of 128, learning rate 0.0003, discount 0.99, GAE 0.95 and clipping 0.2. CPU runs use one Torch thread. Episodes sample navigation/key progression/combat with 30%/40%/30% probabilities. The manifest contains 30 training, 10 validation and 10 test layouts; combat variants share navigation geometry within their own split.')
p('Training rewards: exit +1, death -1, step -0.002, first key +0.25, first passable locked door after key +0.5, kill +0.3, damage -0.002 per point, and first 64-unit cell visit +0.01. Privileged state is used only for training rewards and evaluation. Event bonuses are paid once; raw rewards and components are logged separately.')
h('USP: inspectable learning, evidence-led simplicity')
p('We implemented both a GRU and key-conditioned external memory, then retained the simpler MLP when results did not support added complexity. Under the earlier 102,400-decision progression budget, validation exits were MLP 22/120, GRU 4/120 and custom memory 0/120. These are earlier-budget comparisons, not equal-budget comparisons with the final mixed agent. The contribution is a reproducible pipeline with explicit failure telemetry, not an unsupported novelty claim.')
story.append(PageBreak())
story.append(Paragraph('Measured results & limits',styles['Title']))
p('Checkpoint selection is frozen before final test evaluation. The rule maximizes the mean of the three validation task-family exit rates; ties use combat kills/episode, fewer decisions and earlier checkpoints. Each map has 20 episodes: environment seeds 0-19, policy seeds 1000-1019, cap 2,100 ticks.')
sel=r['selected'];test=r['test']['tasks'];random=r['random_test']['tasks']
table([['Task','Selected validation','Final test exits','Random test exits']]+[[name.title(),f"{sel['tasks'][name]['exit']}/{sel['tasks'][name]['episodes']}",f"{test[name]['exit']}/{test[name]['episodes']}",f"{random[name]['exit']}/{random[name]['episodes']}"] for name in ['navigation','progression','combat']])
h('Combat and progression detail')
p(f"Final combat test: {test['combat']['kills']} total kills, {test['combat']['death']} deaths and {test['combat']['timeout']} timeouts across {test['combat']['episodes']} episodes. Progression test: {test['progression']['key']} episodes collected a key; {test['progression']['passable_after_key']} made a locked door passable after collection. Exits remain the primary reported outcome.")
o=r['official_map'];p(f"Official integration map two_rooms.wad: {o['exits']}/20 exits, {o['deaths']} deaths, {o['timeouts']} timeouts, {o['kills_mean']:.2f} mean kills. This map is separate from our generated benchmark. A validator pass establishes integration and timing, not gameplay competence.")
h('Reproducibility and deployment')
p(f"Selected training checkpoint: {sel['decisions']:,} decisions; see docs/project/FINAL_RESULTS.json for provenance, every episode and candidates. SHA-256: <font size='8'>{r['submission_sha256']}</font>. Final weights live under submission/ and use Git LFS. Source includes generators, manifests with map hashes, PPO, event tests, evaluation, resume state handling and demo commands. Inference depends only on the official base-image packages.")
h('What the evidence does not establish')
p('Generated maps are small and do not cover the full hidden gauntlet, multiple locks, ambushes or scarce-resource combat. Navigation test geometries were inspected in earlier experiments; combat variants inherit that historical exposure. Progression test policy evaluation was reserved for the final freeze. Repeated validation selection is optimistic. The MLP has no explicit memory, so backtracking success is not proof of remembering doors. The official judge score and unseen-map performance remain unknown.')
p('Resume restores optimizer and random states but starts a fresh game episode. Time-limit value estimates use the wrapper\'s stale terminal observation as a documented approximation. Local macOS checks do not establish Linux RAM limits or CUDA equivalence. Review the saved validator output for the exact checks performed.')
def footer(c,doc):
 c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#66788a'));c.drawString(46,28,'Yugaantar2.0 | PPO agent | Measured results, not hidden-evaluator claims');c.drawRightString(550,28,str(doc.page))
SimpleDocTemplate(str(out/'project_brief.pdf'),pagesize=(595.28,841.89),rightMargin=46,leftMargin=46,topMargin=38,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
print(out/'project_brief.pdf')
