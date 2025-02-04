"""
GRAPHICAL USER INTERFACE FOR THE OPTIMISATION TOOL

@author: sergi
"""

import os
import tkinter as tk


# GLOBAL VARIABLES
# TODO: reduce amount of global variables
devices_dict = dict()

blocks_dict = {} #{tag:{'type':str}, ...}
arcs_dict = {} #{tag:{'from':BlockName.port_Name,'to':BlockName.port_Name}, ...}
last_block_id = {d: 0 for d in devices_dict}
last_arc_id = 0
block_from = ''
block_to = ''

init_pos = [0,0]
is_drawing = False


# FUNCTIONS DEFINITION

# Read and return the available devices from the main/Devices folder
def read_devices():
    dev_dict = dict()
    dev_name = ""
    for n in os.listdir("../Devices"):
        if "_" not in n:
            with open("../Devices/"+n,'r') as f:
                for l in f:
                    # device name
                    if l[0:3] == "def":
                        dev_name = l[4:].split("(")[0] # def DeviceName(args):
                        dev_dict[dev_name] = {'Ports':[]}
                    # device ports
                    if "Port(" in l and dev_name!="": # b.port_name = Port(intialize=...):
                        dev_dict[dev_name]['Ports'].append(l.split(".")[1].split()[0])
    return dev_dict

# Binds canvas functions to cursor event depending on selection
def set_current_canvas_function(func):
    global is_drawing, blocks_dict
    
    # unbind everything
    canvas.unbind('<Button-1>')
    canvas.unbind('<ButtonRelease-1>')
    is_drawing = False
    
    # place arc
    if func == 'sel':
        canvas.config(cursor='top_left_arrow')
        canvas.bind('<Button-1>', select_item)
        
    elif func == 'arc':
        canvas.config(cursor='cross')
        canvas.bind('<Button-1>', draw_arc)
    
    # place block
    elif func == 'block':
        canvas.config(cursor='dotbox')
        canvas.bind('<ButtonRelease-1>', draw_block)

# Selects an item on canvas
def select_item(event):
    item = canvas.gettags("current")
    if len(item)>0:
        item = item[0].replace("_txt","")
        item_selection.set(item)
        item_selection_label.config(text=item)

# Place an arc on canvas
# TODO: (issue #28) fix error when selecting something that is not a Block -> do not allow
def draw_arc(event):
    global is_drawing, init_pos, arcs_dict, last_arc_id, block_from, block_to
    
    if is_drawing: # end
        
        arc_name = "arc" + str(last_arc_id)
        last_arc_id = last_arc_id+1
    
        canvas.create_line(init_pos[0], init_pos[1], event.x, event.y, arrow=tk.LAST, tag=arc_name, activefill="red")
        txt = canvas.create_text((event.x+init_pos[0])/2, (event.y+init_pos[1])/2, tag=arc_name+'_txt')
        canvas.insert(txt, 10, arc_name)
        
        block_to = canvas.gettags("current")[0].replace("_txt","")
        block_to = show_dialog_ports(block_to)
        
        init_pos = [0,0]
        is_drawing = False
        
        arcs_dict[arc_name]={'from':block_from, 'to':block_to}
        
    else: # start
        is_drawing = True
        init_pos[0], init_pos[1] = event.x, event.y
        block_from = canvas.gettags("current")[0].replace("_txt","")
        block_from = show_dialog_ports(block_from)

def show_dialog_ports(block_name):
    device_type = blocks_dict[block_name]['type']
    port_name = tk.simpledialog.askstring(title=block_name, parent=window, 
                                        prompt=str(devices_dict[device_type]['Ports']))
    block_name = block_name+'.'+port_name
    return block_name

# Place a block on canvas   
def draw_block(event):
    global code_blocks, blocks_dict, last_block_id
    
    block_name = devices_menu_selection.get() + str(last_block_id[devices_menu_selection.get()])
    last_block_id[devices_menu_selection.get()] = last_block_id[devices_menu_selection.get()]+1
    
    txt = canvas.create_text(event.x, event.y, tag=block_name+'_txt')
    canvas.insert(txt, 10, block_name)
    canvas.create_rectangle(event.x-10, event.y-10, event.x+10, event.y+10, tag=block_name, activefill="red")
    
    # TODO: (issue #21) add full info to dict
    blocks_dict[block_name]={'type':devices_menu_selection.get()}
        
    
# Delete block/arc
# TODO: (issue #25) better delete for arcs and blocks
# TODO: (issue #26) error when no blocks or arcs left
# TODO: (issue #27) error when last block/arc when setting var, since list is empty --> delete_menu2_selection.set(arcs_list[0])
def delete_item(item_name):
    global blocks_dict, arcs_dict
    
    canvas.delete(item_name)
    canvas.delete(item_name+'_txt')
    if item_name in blocks_dict:
        blocks_dict.pop(item_name)
    elif item_name in arcs_dict:
        arcs_dict.pop(item_name)
        
    item_selection.set(None)
    item_selection_label.config(text='')
    
# write the code of the current system
def write_code():
    code_blocks = "# model\nm = pyo.ConcreteModel()\n\n#BLOCKS\n"
    code_arcs = "#CONNECTIONS\n"
    
    for b in blocks_dict:
        # TODO: (issue #23) full data, init and t
        code = "m."+b+" = pyo.Block()\n"
        code = code + "data = {}\n"
        code = code + "init = {}\n"
        code = code + b+"("+b+", m.t, data, init)\n"
        code = code + "\n"
        
        code_blocks = code_blocks + code

    for a in arcs_dict:
        code = "m.{}=Arc(ports=(m.{}, m.{}), directed=True)\n".format(a, arcs_dict[a]['from'], arcs_dict[a]['to'])
        code_arcs = code_arcs + code
    
    code_arcs = code_arcs + "pyo.TransformationFactory(\"network.expand_arcs\").apply_to(m)\n"

    print("Writing to file ...")
    with open("model_from_gui.py",'w') as f:
        f.write(code_blocks)
        f.write(code_arcs)
    print("DONE")

# === TKINTER WINDOW === 

devices_dict = read_devices()
last_block_id = {d: 0 for d in devices_dict}
init_pos = [0,0]
is_drawing = False

window = tk.Tk()

window.title("System")
window.geometry("800x700")
window.resizable(False,False)

# TOOLBOX
# Arcs and Blocks
frame_tool_left = tk.Frame(window, width=400, height=100)
frame_tool_left.grid(row=0, column=0, sticky=tk.NW)

# -- select
sel_btn = tk.Button(frame_tool_left, text='Select', command=lambda: set_current_canvas_function('sel'))
sel_btn.grid(row=0,column=0)

# -- arcs
arc_btn = tk.Button(frame_tool_left, text='Arc', command=lambda: set_current_canvas_function('arc'))
arc_btn.grid(row=0,column=1)

# -- devices
device_btn = tk.Button(frame_tool_left, text='Device', command=lambda: set_current_canvas_function('block'))
device_btn.grid(row=0,column=2)
    
devices_menu_selection = tk.StringVar()
devices_menu_selection.set(list(devices_dict.keys())[0])
devices_menu = tk.OptionMenu(frame_tool_left, devices_menu_selection, *list(devices_dict.keys()))
devices_menu.grid(row=0,column=3)

# Selection and Delete
frame_tool_right = tk.Frame(window, width=400, height=100)
frame_tool_right.grid(row=0, column=0, sticky=tk.NE)

item_selection = tk.StringVar()
item_selection_label = tk.Label(frame_tool_right, text='')
item_selection_label.grid(row=0,column=0)

del_btn = tk.Button(frame_tool_right, text='Delete',
                    command=lambda: delete_item(item_selection.get()))
del_btn.grid(row=0,column=1)

# CANVAS
canvas_frame = tk.Frame(window, width=800, height=600)
canvas_frame.grid(row=1, column=0)

canvas = tk.Canvas(canvas_frame, width=800, height=600, bg="white")
canvas.grid(row=0,column=0)


# MAIN LOOP
window.mainloop()
write_code()
