import smtplib, os, inspect, sys
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)
from leafmachine2.machine.general_utils import get_cfg_from_full_path

def setup_email(pc):
    try:
        path_cfg_private = os.path.join(parentdir, 'PRIVATE_DATA.yaml')
        cfg_private = get_cfg_from_full_path(path_cfg_private)
        if pc == "pc":
            pwd = cfg_private['gmail_pwd']
        elif pc == "quadro":
            pwd = cfg_private['gmail_pwd_quadro']
        elif pc == "ada":
            pwd = cfg_private['gmail_pwd_ada']
        else:
            pwd = cfg_private['gmail_pwd']
        from_email = cfg_private['gmail_account']
        to_email = cfg_private['gmail_account_to']
        return pwd, from_email, to_email
    except:
        return None, None, None
    
def send_update(path, message, pc="quadro"):
    try:
        working_on = os.path.basename(path)
        subject = f"{message} {working_on}"
        success = send_email(subject, f"Working on dataset {path}", pc)
        if success:
            print('Email notification sent successfully')
        else:
            print('Email notification failed')
    except:
        print('Email notification failed')

def send_email(subject, body, pc="pc"):
    """
    Sends an email with the specified subject and body.
    """
    password, from_email, to_email  = setup_email(pc) 

    if password is None:
        return False
    
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    # Create a multipart message
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    # Attach the body text
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Connect to the SMTP server and send the email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Upgrade the connection to secure
            server.login(from_email, password)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    name = "Quercus"
    message = f"LM2 finished --- {name}"
    
    # Email details
    subject = message
    body = message

    # Send the notification email
    success = send_email(subject, body)
