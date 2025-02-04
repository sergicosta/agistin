"""
GRAPHICAL USER INTERFACE FOR THE OPTIMISATION TOOL

@author: sergi
"""

import os
import tkinter as tk


# GLOBAL VARIABLES

devices_list = read_devices()    

blocks_dict = {} #{tag:{'id':int, 'device':str, 'pos':(x,y,xf,yf)}, ...}
arcs_dict = {} #{tag:{'from':,'to':}, ...}
last_block_id = {d: 0 for d in devices_list}
last_arc_id = 0
blocks_list = ['']
arcs_list = ['']
block_from = ''
block_to = ''

init_pos = [0,0]
is_drawing = False


# FUNCTIONS DEFINITION

# Read and return the available devices from the main/Devices folder
def read_devices():
    dev_list = []
    for n in os.listdir("../Devices"):
        if "_" not in n:
            with open("../Devices/"+n,'r') as f:
                for l in f:
                    if l[0:3] == "def":
                        dev_list.append(l[4:].split("(")[0])
    return dev_list

# Binds canvas functions to cursor event depending on selection
def set_current_canvas_function(func):
    global is_drawing, blocks_dict
    canvas.unbind('<Button-1>')
    canvas.unbind('<ButtonRelease-1>')
    
    if func == 'arc':
        is_drawing = False
        canvas.config(cursor='cross')
        canvas.bind('<Button-1>', draw_arc)
    
    elif func == 'block':
        canvas.config(cursor='dotbox')
        canvas.bind('<ButtonRelease-1>', draw_block)

# Refresh current blocks placed on canvas
def resfresh_blocks_list():
    global blocks_list
    
    blocks_list = list(blocks_dict.keys())
    delete_menu_selection.set(blocks_list[0])
    
    delete_menu["menu"].delete(0, "end")
    
    for b in blocks_list:
        delete_menu["menu"].add_command(label=b, command=tk._setit(delete_menu_selection, b))
        
# Refresh current arcs placed on canvas
def resfresh_arcs_list():
    global arcs_list
    
    arcs_list = list(arcs_dict.keys())
    delete_menu2_selection.set(arcs_list[0])
    
    delete_menu2["menu"].delete(0, "end")
    
    for a in arcs_list:
        delete_menu2["menu"].add_command(label=a, command=tk._setit(delete_menu2_selection, a))
             
# Place an arc on canvas   
def draw_arc(event):
    global is_drawing, init_pos, arcs_dict, last_arc_id, block_from, block_to
    
    if is_drawing: # end
        
        arc_name = "arc" + str(last_arc_id)
        last_arc_id = last_arc_id+1
    
        canvas.create_line(init_pos[0], init_pos[1], event.x, event.y, arrow=tk.LAST, tag=arc_name, activefill="red")
        txt = canvas.create_text((event.x+init_pos[0])/2, (event.y+init_pos[1])/2, tag=arc_name+'_txt')
        canvas.insert(txt, 10, arc_name)
        
        block_to = canvas.gettags("current")[0]
        
        init_pos = [0,0]
        is_drawing = False
        
        # TODO: add full info to dict. From and to should be BlockName.Port
        arcs_dict[arc_name]={'from':block_from, 'to':block_to}
        resfresh_arcs_list()
        
    else: # start
        is_drawing = True
        init_pos[0], init_pos[1] = event.x, event.y
        block_from = canvas.gettags("current")[0]
    
# Place a block on canvas   
def draw_block(event):
    global code_blocks, blocks_dict, last_block_id
    
    block_name = devices_menu_selection.get() + str(last_block_id[devices_menu_selection.get()])
    last_block_id[devices_menu_selection.get()] = last_block_id[devices_menu_selection.get()]+1
    
    txt = canvas.create_text(event.x, event.y, tag='block_'+block_name+'_txt')
    canvas.insert(txt, 10, block_name)
    canvas.create_rectangle(event.x-10, event.y-10, event.x+10, event.y+10, tag='block_'+block_name, activefill="red")
    
    # TODO: add full info to dict
    blocks_dict[block_name]={}
        
    resfresh_blocks_list()

# Delete block/arc
# TODO: error when no blocks or arcs left
# TODO: error when last block/arc when setting var, since list is empty --> delete_menu2_selection.set(arcs_list[0])
def delete_block(block_name):
    global blocks_dict
    
    canvas.delete("block_"+block_name)
    canvas.delete("block_"+block_name+'_txt')
    blocks_dict.pop(block_name)

    resfresh_blocks_list()
def delete_arc(arc_name):
    global arcs_dict
    
    canvas.delete(arc_name)
    canvas.delete(arc_name+'_txt')
    arcs_dict.pop(arc_name)

    resfresh_arcs_list()
    
# write the code of the current system
def write_code():
    code_blocks = "# model\nm = pyo.ConcreteModel()\n\n#BLOCKS\n"
    code_arcs = "#CONNECTIONS\n"
    
    for b in blocks_dict:
        # TODO: full data, init and t
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

window = tk.Tk()

window.title("System")
window.geometry("800x700")
window.resizable(False,False)

# TOOLBOX
# Arcs and Blocks
frame_tool_left = tk.Frame(window, width=400, height=100)
frame_tool_left.grid(row=0, column=0, sticky=tk.NW)

# -- arcs
arc_btn = tk.Button(frame_tool_left, text='Arc', command=lambda: set_current_canvas_function('arc'))
arc_btn.grid(row=0,column=0)

# -- devices
device_btn = tk.Button(frame_tool_left, text='Device', command=lambda: set_current_canvas_function('block'))
device_btn.grid(row=0,column=1)
    
devices_menu_selection = tk.StringVar()
devices_menu_selection.set(devices_list[0])
devices_menu = tk.OptionMenu(frame_tool_left, devices_menu_selection, *devices_list)
devices_menu.grid(row=0,column=2)

# Delete
frame_tool_right = tk.Frame(window, width=400, height=100)
frame_tool_right.grid(row=0, column=0, sticky=tk.NE)

# -- blocks
delete_menu_selection = tk.StringVar()
delete_menu = tk.OptionMenu(frame_tool_right, delete_menu_selection, *blocks_list)
delete_menu.grid(row=0,column=0)

del_btn = tk.Button(frame_tool_right, text='Delete Block', width=8,
                    command=lambda: delete_block(delete_menu_selection.get()))
del_btn.grid(row=0,column=1)

# -- arcs
delete_menu2_selection = tk.StringVar()
delete_menu2 = tk.OptionMenu(frame_tool_right, delete_menu2_selection, *arcs_list)
delete_menu2.grid(row=1,column=0)

del_btn2 = tk.Button(frame_tool_right, text='Delete Arc', width=8,
                     command=lambda: delete_arc(delete_menu2_selection.get()))
del_btn2.grid(row=1,column=1)

# CANVAS
canvas_frame = tk.Frame(window, width=800, height=600)
canvas_frame.grid(row=1, column=0)

canvas = tk.Canvas(canvas_frame, width=800, height=600, bg="white")
canvas.grid(row=0,column=0)


# MAIN LOOP
window.mainloop()
write_code()
